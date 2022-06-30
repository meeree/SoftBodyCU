#pragma once
#include "util.cuh"
#include "helper_math.cuh"
#include <fstream>
#include "graphics.h"
#include <algorithm>
#include <unordered_set>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include "polygonizer.h"
#include "density_sampler.h"
#include "delaunator.hpp"

inline __global__ void handle_springs_kernel(int const N,
        int const* row_ptr, int const* col_ind, 
        float const* strengths, float const* lengths, 
        float3 const* X, float3* F)
{
	for(int pre = blockIdx.x * blockDim.x + threadIdx.x; pre < N; pre += gridDim.x * blockDim.x)
    {
        // Sum up spring force for all downstream springs.
        float3 x_pre = X[pre];
        float3 F_pre = make_float3(0.0);
        for(int post = row_ptr[pre]; post < row_ptr[pre+1]; ++post)
        {
            float3 x_post = X[col_ind[post]]; 
            float r = length(x_pre - x_post);
            float f = strengths[post] * (r - lengths[post]);
            F_pre += f * (x_post - x_pre) / r;
        }
        F[pre] = F_pre;
	}
}

inline __global__ void euler_kernel(int const N,
    float3* X, float3* V, float3* A, float3 const* F, float const* M_inv)
{
    float dt = 0.02;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += gridDim.x * blockDim.x)
    {
        float3 f = F[tid] - make_float3(0.0, 0.0001, 0.0); // Gravity
        A[tid] = f * M_inv[tid];
        V[tid] += A[tid] * dt;
        if (X[tid].y < -3.5 && V[tid].y < 0)
            V[tid].y = -V[tid].y;

        X[tid] += V[tid] * dt;
    }
}

struct Springs_CPU;
struct Springs
{
    // Adjacency List Format. 
	thrust::device_vector<int> row_ptr;
    thrust::device_vector<int> col_ind;
    thrust::device_vector<float> strengths;
    thrust::device_vector<float> lengths;
    Springs() {}
    Springs(Springs_CPU const& rhs);
};
struct Springs_CPU
{
	thrust::host_vector<int> row_ptr;
    thrust::host_vector<int> col_ind;
    thrust::host_vector<float> strengths;
    thrust::host_vector<float> lengths;
    Springs_CPU() {}
    Springs_CPU (Springs const& rhs) // gpu -> cpu
    {
        row_ptr = rhs.row_ptr;
        col_ind = rhs.col_ind;
        strengths = rhs.strengths;
        lengths = rhs.lengths;
    }
};
Springs::Springs (Springs_CPU const& rhs) // cpu -> gpu
{
    row_ptr = rhs.row_ptr;
    col_ind = rhs.col_ind;
    strengths = rhs.strengths;
    lengths = rhs.lengths;
}

struct System_CPU;
struct System
{
    thrust::device_vector<float3> X, V, A, F; // Position, Velocity, Acceleration, Force.
    thrust::device_vector<float> M_inv; // Inverse mass.
    Springs springs;
    System() {}
    System(System_CPU const& rhs);
};
struct System_CPU
{
    thrust::host_vector<float3> X, V, A, F;
    thrust::host_vector<float> M_inv; 
    Springs_CPU springs;
    System_CPU() {}
    System_CPU (System const& rhs) // cpu -> gpu
    {
        X = rhs.X; V = rhs.V; A = rhs.A; F = rhs.F; M_inv = rhs.M_inv;
        springs = rhs.springs;
    }
};
System::System (System_CPU const& rhs) // gpu -> cpu
{
    X = rhs.X; V = rhs.V; A = rhs.A; F = rhs.F; M_inv = rhs.M_inv;
    springs = rhs.springs;
}

void handle_springs(System& sys)
{
    handle_springs_kernel<<<NBLOCKS, NTHREADS>>>(
        sys.X.size(), 
        RAW(sys.springs.row_ptr), RAW(sys.springs.col_ind),
        RAW(sys.springs.strengths), RAW(sys.springs.lengths),
        RAW(sys.X), RAW(sys.F)
    );
    gpuErrchk( cudaPeekAtLastError() ); // Check for errors.
	gpuErrchk( cudaDeviceSynchronize() ); // Synchronize kernel.
}

void euler_step(System& sys)
{
    euler_kernel<<<NBLOCKS, NTHREADS>>>(
        sys.X.size(), 
        RAW(sys.X), RAW(sys.V), RAW(sys.A),
        RAW(sys.F), RAW(sys.M_inv)
    );
    gpuErrchk( cudaPeekAtLastError() ); // Check for errors.
	gpuErrchk( cudaDeviceSynchronize() ); // Synchronize kernel.
}

thrust::host_vector<float> linspace(float a, float b, int n)
{
    thrust::host_vector<float> res(n);
    for(int i = 0; i < n; ++i)
        res[i] = a + (b - a) * (i / float(n-1));
    return res;
}

// Setup grid of points and springs.
System grid2d(int m, int n, float a, float b)
{
    System sys;

    // setup positions and state variables.
    // ----Y  (Y indexes rows, X indexes columns)
    // |
    // |
    // X 
    auto xs = linspace(-a, a, m), ys = linspace(-b, b, n);
    thrust::host_vector<float3> X(m * n);
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
            X[i*n + j] = make_float3(xs[i], 0.0, ys[j]);
    sys.X = X;
    sys.M_inv = thrust::device_vector<float>(m * n, 1.0);
    sys.V = thrust::device_vector<float3>(m * n, make_float3(0.0));
    sys.A = sys.V;
    sys.F = sys.V;

    // Setup springs.
    thrust::host_vector<thrust::host_vector<int>> downstream_inds(m*n);
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            // Verical springs.
            if(i < m-1)
            {
                downstream_inds[i*n+j].push_back((i+1)*n+j);
                downstream_inds[(i+1)*n+j].push_back(i*n+j);
            }
            // Horizontal springs.
            if(j < n-1)
            {
                downstream_inds[i*n+j].push_back(i*n+j+1);
                downstream_inds[i*n+j+1].push_back(i*n+j);
            }
        }
    }
    thrust::host_vector<int> row_ptr(m*n+1,0), col_ind;
    for(int i = 0; i < m*n; ++i)
    {
        row_ptr[i+1] = downstream_inds[i].size() + row_ptr[i];
        for(int val: downstream_inds[i])
            col_ind.push_back(val);
    }
    int n_springs = col_ind.size();
    std::cout << "Grid: generated " << n_springs / 2 << " bi-directional springs." << std::endl;
    sys.springs.row_ptr = row_ptr;
    sys.springs.col_ind = col_ind;
    sys.springs.strengths = thrust::device_vector<float>(n_springs, 0.1);

    // TODO: Move length setup to its own generic function.
    thrust::host_vector<float> lengths(n_springs);
    for(int i = 0; i < m*n; ++i)
    {
        int start = row_ptr[i], end = row_ptr[i+1];
        for(int s = start; s < end; ++s)
        {
            float3 x1 = X[i], x2 = X[col_ind[s]];
            lengths[s] = length(x1 - x2);
        }
    }
    sys.springs.lengths = lengths;
    return sys;
}

System grid3d(Grid const& grid)
{
    int m = grid.mDim[0], n = grid.mDim[1], p = grid.mDim[2];
    System sys;

    auto xs = linspace(grid.mOrigin[0], grid.mOrigin[0] + grid.mSize[0], m), 
         ys = linspace(grid.mOrigin[1], grid.mOrigin[1] + grid.mSize[1], n), 
         zs = linspace(grid.mOrigin[2], grid.mOrigin[2] + grid.mSize[2], p);
    thrust::host_vector<float3> X(m * n * p);
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
            for(int k = 0; k < p; ++k)
                X[i*n*p + j*n + k] = make_float3(xs[i], ys[j], zs[k]);
    sys.X = X;
    sys.M_inv = thrust::device_vector<float>(m * n * p, 1.0);
    sys.V = thrust::device_vector<float3>(m * n * p, make_float3(0.0));
    sys.A = sys.V;
    sys.F = sys.V;

    // Setup springs.
    thrust::host_vector<thrust::host_vector<int>> downstream_inds(m*n*p);
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            for (int k = 0; k < p; ++k)
            {
                // Verical springs.
                if (i < m - 1)
                {
                    downstream_inds[i*n*m + j*n + k].push_back((i + 1)*n*m + j*n + k);
                    downstream_inds[(i + 1)*n*m + j*n + k].push_back(i*n*m + j*n + k);
                }
                // Horizontal springs.
                if (j < n - 1)
                {
                    downstream_inds[i*n*m + j*n + k].push_back(i*n*m + (j + 1)*n + k);
                    downstream_inds[i*n*m + (j + 1)*n + k].push_back(i*n*m + j*n + k);
                }

                // Deep springs.
                if (j < p - 1)
                {
                    downstream_inds[i*n*m + j*n + k].push_back(i*n*m + j*n + k + 1);
                    downstream_inds[i*n*m + j*n + k + 1].push_back(i*n*m + j*n + k);
                }

                // X-Y diagonal springs.
                if (i < m - 1 && j < n - 1)
                {
                    downstream_inds[i * n * m + j * n + k].push_back((i + 1) * n * m + (j + 1) * n + k);
                    downstream_inds[(i + 1) * n * m + (j + 1) * n + k].push_back(i * n * m + j * n + k);
                }

                // X-Z diagonal springs.
                if (i < m - 1 && k < p - 1)
                {
                    downstream_inds[i * n * m + j * n + k].push_back((i + 1) * n * m + j * n + k + 1);
                    downstream_inds[(i + 1) * n * m + j * n + k + 1].push_back(i * n * m + j * n + k);
                }

                // Y-Z diagonal springs.
                if (j < n - 1 && k < p - 1)
                {
                    downstream_inds[i * n * m + j * n + k].push_back(i * n * m + (j + 1) * n + k + 1);
                    downstream_inds[i * n * m + (j + 1) * n + k + 1].push_back(i * n * m + j * n + k);
                }
            }
        }
    }
    thrust::host_vector<int> row_ptr(m*n*p+1,0), col_ind;
    for(int i = 0; i < m*n*p; ++i)
    {
        row_ptr[i+1] = downstream_inds[i].size() + row_ptr[i];
        for(int val: downstream_inds[i])
            col_ind.push_back(val);
    }
    int n_springs = col_ind.size();
    std::cout << "Grid: generated " << n_springs / 2 << " bi-directional springs." << std::endl;
    sys.springs.row_ptr = row_ptr;
    sys.springs.col_ind = col_ind;
    sys.springs.strengths = thrust::device_vector<float>(n_springs, 0.1);

    // TODO: Move length setup to its own generic function.
    thrust::host_vector<float> lengths(n_springs, xs[1] - xs[0]);
    for(int i = 0; i < row_ptr.size()-1; ++i)
    {
        int start = row_ptr[i], end = row_ptr[i+1];
        for(int s = start; s < end; ++s)
        {
            float3 x1 = X[i], x2 = X[col_ind[s]];
            lengths[s] = length(x1 - x2);
        }
    }
    sys.springs.lengths = lengths;
    return sys;
}

#define F2G(v) (glm::vec3(v.x,v.y,v.z))

System implicit_surface(Grid const& grid, DensitySampler const& sampler, thrust::host_vector<float3>& normals)
{
    System ball_sys;
    Springs_CPU ball_springs_cpu;
    { // Open a new scope so sys is deallocated on gpu before we send new springs cpu -> gpu.
        System sys = grid3d(grid);

        thrust::host_vector<float3> X_cpu = sys.X;
        thrust::host_vector<int> inds;
        for (int i = 0; i < sys.X.size(); ++i)
        {
            float3 v = X_cpu[i];
            if(sampler.sample(glm::vec3(v.x, v.y, v.z)) < 0.0)
                inds.push_back(i);
        }

        int m = grid.mDim[0], n = grid.mDim[1], p = grid.mDim[2];

        // Unordered_set is used for faster lookup.
        std::unordered_set<int> inds_lookup(inds.begin(), inds.end());
        thrust::host_vector<int> new_inds(sys.X.size()); // Lookup for new indices after culling.
        for (int i = 0; i < new_inds.size(); ++i)
        {
            new_inds[i] = i > 0 ? new_inds[i - 1] : -1;
            if (inds_lookup.find(i) != inds_lookup.end())
                new_inds[i] += 1;
        }

        // Copy positions that are within radius.
        thrust::host_vector<float3> new_X(inds.size());
        for (int i = 0; i < inds.size(); ++i)
            new_X[i] = X_cpu[inds[i]];
        ball_sys.X = new_X; // cpu -> gpu.
        ball_sys.V.resize(inds.size(), make_float3(0.0));
        ball_sys.A.resize(inds.size(), make_float3(0.0));
        ball_sys.F.resize(inds.size(), make_float3(0.0));
        ball_sys.M_inv.resize(inds.size(), 1.0);

        normals.resize(inds.size());
        for(int i = 0; i < inds.size(); ++i)
        {
            glm::vec3 v{ F2G(X_cpu[inds[i]]) };
            glm::vec3 pxp{ v + glm::vec3(1.0f, 0.0f, 0.0f) * 1e-6f};
            glm::vec3 pxn{ v - glm::vec3(1.0f, 0.0f, 0.0f) * 1e-6f};
            glm::vec3 pyp{ v + glm::vec3(0.0f, 1.0f, 0.0f) * 1e-6f};
            glm::vec3 pyn{ v - glm::vec3(0.0f, 1.0f, 0.0f) * 1e-6f};
            glm::vec3 pzp{ v + glm::vec3(0.0f, 0.0f, 1.0f) * 1e-6f};
            glm::vec3 pzn{ v - glm::vec3(0.0f, 0.0f, 1.0f) * 1e-6f};
            normals[i] =
                make_float3(
                    sampler.sample(pxp) - sampler.sample(pxn),
                    sampler.sample(pyp) - sampler.sample(pyn),
                    sampler.sample(pzp) - sampler.sample(pzn));
            normals[i] /= -length(normals[i]);
        }

        // Cull springs outside radius. 
        Springs_CPU springs_cpu = sys.springs; // gpu -> cpu
        ball_springs_cpu.row_ptr.push_back(0);
        for (int i = 0; i < sys.X.size(); ++i)
        {
            if (inds_lookup.find(i) != inds_lookup.end()) // Cull rows
            {
                for (int j = springs_cpu.row_ptr[i]; j < springs_cpu.row_ptr[i + 1]; ++j)
                {
                    int ind = springs_cpu.col_ind[j];
                    if (inds_lookup.find(ind) != inds_lookup.end()) // Cull columns
                    {
                        ball_springs_cpu.col_ind.push_back(new_inds[ind]);
                        ball_springs_cpu.lengths.push_back(springs_cpu.lengths[j]);
                        ball_springs_cpu.strengths.push_back(springs_cpu.strengths[j]);
                    }
                }
                ball_springs_cpu.row_ptr.push_back(ball_springs_cpu.col_ind.size());
            }
        }
    }
    Springs springs_gpu(ball_springs_cpu); // cpu -> gpu
    ball_sys.springs = std::move(springs_gpu);
    return ball_sys;
}

// For rendering springs.
std::vector<int> springs_to_lines(Springs const& springs)
{
    std::vector<int> lines;
    Springs_CPU sp_cpu = springs; // gpu -> cpu.
    for (int i = 1; i < sp_cpu.row_ptr.size(); ++i)
    {
        for (int j = sp_cpu.row_ptr[i - 1]; j < sp_cpu.row_ptr[i]; ++j)
        {
            lines.push_back(i - 1);
            lines.push_back(sp_cpu.col_ind[j]);
        }
    }
    return lines;
}

System delaunay_example()
{
    srand(time(NULL));
    int N = 1000;
    /* x0, y0, x1, y1, ... */
    std::vector<double> coords(2 * N);
    for (double& coord : coords)
        coord = -2 + 4 * (rand() % 10000) / (10000.0 - 1);

    // Cull points outside circle
    std::vector<int> cull_inds;
    for (int i = 0; i < coords.size() / 2; ++i)
    {
        double x = coords[2 * i], y = coords[2 * i + 1];
        if (x * x + y * y > 4)
        {
            coords.erase(coords.begin() + 2 * i);
            coords.erase(coords.begin() + 2 * i + 1);
            i -= 1;
        }
    }
    N = coords.size() / 2;

    //triangulation happens here
    delaunator::Delaunator d(coords);

    std::vector<float> coords_3d;
    thrust::host_vector<float3> x_cpu;
    for (int i = 0; i < coords.size() / 2; ++i)
    {
        coords_3d.push_back(0);
        coords_3d.push_back(coords[2 * i + 1]);
        coords_3d.push_back(coords[2 * i + 0]);
        x_cpu.push_back(make_float3(0, coords[2 * i + 1], coords[2 * i]));
    }
    System sys;
    sys.X = x_cpu;
    sys.V = thrust::device_vector<float3>(N, make_float3(0.0));
    sys.A = thrust::device_vector<float3>(N, make_float3(0.0));
    sys.F = thrust::device_vector<float3>(N, make_float3(0.0));
    sys.M_inv = thrust::device_vector<float>(N, 1.0);

    std::vector<std::pair<int, int>> line_inds(d.triangles.size());
    for (int i = 0; i < d.triangles.size() / 3; ++i)
    {
        line_inds[3 * i] = { d.triangles[3 * i], d.triangles[3 * i + 1] };
        line_inds[3 * i + 1] = { d.triangles[3 * i + 1], d.triangles[3 * i + 2] };
        line_inds[3 * i + 2] = { d.triangles[3 * i + 2], d.triangles[3 * i] };
    }
    // Make it  so that line pairs have ind1 <= ind2.
    for (auto& ind : line_inds)
        if (ind.first > ind.second)
            std::swap(ind.first, ind.second);

    // Remove duplicate lines.
    line_inds.erase(std::unique(line_inds.begin(), line_inds.end()), line_inds.end());

    thrust::host_vector<thrust::host_vector<int>> downstream(N);
    for (int i = 0; i < line_inds.size(); ++i)
    {
        int ind1 = line_inds[i].first, ind2 = line_inds[i].second;
        downstream[ind1].push_back(ind2);
        downstream[ind2].push_back(ind1);
        float len = length(x_cpu[ind1] - x_cpu[ind2]);
    }

    sys.springs.row_ptr.push_back(0);
    for (int i = 0; i < downstream.size(); ++i)
        sys.springs.row_ptr.push_back(downstream[i].size() + sys.springs.row_ptr.back());
    int nnz = sys.springs.row_ptr.back();
    sys.springs.col_ind.resize(nnz);
    sys.springs.strengths.resize(nnz, 0.1);
    thrust::host_vector<float> lengths(nnz);
    for (int i = 0; i < downstream.size(); ++i)
    {
        thrust::copy(downstream[i].begin(), downstream[i].end(), sys.springs.col_ind.begin() + sys.springs.row_ptr[i]);
        for (int j = sys.springs.row_ptr[i]; j < sys.springs.row_ptr[i + 1]; ++j)
        {
            int ind = downstream[i][j - sys.springs.row_ptr[i]];
            auto x1 = x_cpu[i], x2 = x_cpu[ind];
            lengths[j] = length(x1 - x2);
        }
    }
    sys.springs.lengths = lengths;
    return sys;
}

int main ()
{
    Grid grid{ {10, 10, 10}, {4.0, 4.0, 4.0}, {-2.0, -2.0, -2.0} };
    DensitySampler sampler;
    thrust::host_vector<float3> normals;
    System sys = implicit_surface(grid, sampler, normals);
    MarchingCubes marcher;

    std::vector<int> tri_inds;
    {
        // Polygonize SDF.
        std::vector<Vertex> full_verts;
        marcher.polygonise(grid, 0.0, full_verts, sampler);
        tri_inds.resize(full_verts.size());

        // Get indices of nearest points.
        thrust::device_vector<float> diffs(sys.X.size());
        for (int i = 0; i < full_verts.size(); ++i)
        {
            float3 v = make_float3(full_verts[i].mPosition.x, full_verts[i].mPosition.y, full_verts[i].mPosition.z);
            thrust::transform(sys.X.begin(), sys.X.end(), diffs.begin(),
                [v] __host__ __device__ (float3 x) {return length(v - x); });

            // Get minimum distance to point.
            int min_ind = thrust::min_element(diffs.begin(), diffs.end()) - diffs.begin();
            tri_inds[i] = min_ind;
        }

        // Set normals based on average triangle normal at point.
        thrust::host_vector<int> incs(normals.size(), 0);
        normals = thrust::host_vector<float3>(sys.X.size(), make_float3(0.0));
        for(int i = 0; i < tri_inds.size(); ++i)
        {
            auto const& nm = full_verts[i].mNormal;
            normals[tri_inds[i]] += make_float3(nm.x, nm.y, nm.z);
            incs[tri_inds[i]] += 1;
        }
        for (int i = 0; i < normals.size(); ++i)
        {
            // Average over triangles adjacent to a vertex..
            if (incs[i] > 0)
            {
                normals[i] /= incs[i];
                normals[i] = normalize(normals[i]);
            }
        }
    }
//    // Fix points
//    auto it = thrust::min_element(sys.X.begin(), sys.X.end(),
//        [] __host__ __device__(float3 v1, float3 v2) {
//        return v1.z < v2.z;
//    });
//    float min_z = ((float3)*it).y;
//    thrust::transform(sys.X.begin(), sys.X.end(), sys.M_inv.begin(),
//        [min_z] __host__ __device__(float3 v) {
//        return fabs(v.z - min_z) < 0.5 ? 0.0 : 1.0;
//    });
//
    sys = delaunay_example();
    thrust::fill(sys.springs.strengths.begin(), sys.springs.strengths.end(), 5.0);

    int N = sys.X.size();
    thrust::host_vector<float3> X_host(sys.X), V_host(sys.V);
    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, (float*)X_host.data(), nullptr, N, N);

    std::vector<int> lines = springs_to_lines(sys.springs);
    Graphics::add_lines(lines.data(), lines.size() / 2);
    Graphics::add_triangles(tri_inds.data(), tri_inds.size() / 3);

    for (int i = 0; i < 10; ++i)
    {
        int N_steps = 20000 / 0.1;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        thrust::host_vector<thrust::device_vector<float3>> Xs(N_steps, sys.X);
        thrust::host_vector<thrust::device_vector<float3>> Vs(N_steps, sys.V);

        cudaEventRecord(start);
        for (int n = 0; n < N_steps; ++n)
        {
            Xs[n] = sys.X;
            Vs[n] = sys.V;
            handle_springs(sys);
            euler_step(sys);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Simulation time: " << milliseconds << "ms" << std::endl;

        float max_vel = 0.0, min_vel = 1000000.0;
        for (auto& arr : Vs)
        {
            thrust::host_vector<float3> arr_host(arr);
            for (auto& val : arr_host)
            {
                if (length(val) > max_vel)
                    max_vel = length(val);
                if (length(val) < min_vel)
                    min_vel = length(val);
            }
        }

        bool pause = false;
        for (int n = 0; n < N_steps; n += 100)
        {
            X_host = Xs[n];
            V_host = Vs[n];
            std::transform(V_host.begin(), V_host.end(), graphics_voltage_handle,
                [&](float3 const& v) {return (length(v) - min_vel) / (max_vel - min_vel); });
            Graphics::update_positions((float*)X_host.data());
            if (!Graphics::render(graphics_voltage_handle, nullptr, 0, 0, pause))
                return 0; // DONE!
        }
    }
}
