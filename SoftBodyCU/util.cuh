#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

#define NTHREADS 512 
#define NBLOCKS 32

// Cast thrust vector to raw pointer for kernel call.
template<class T>
inline T* RAW(thrust::device_vector<T>& v)
{
    return thrust::raw_pointer_cast(v.data());
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Get indices of thrust vector that match a query. All is done on GPU.
template<class T, typename QueryFun>
thrust::device_vector<int> thrust_query(thrust::device_vector<T> const& vec, QueryFun query)
{
    thrust::device_vector<int> indices(vec.size());
    thrust::device_vector<int>::iterator end = thrust::copy_if(thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(vec.size()),
        vec.begin(),
        indices.begin(),
        query);
    int size = end - indices.begin();
    indices.resize(size); // Remove unused indices.
    return indices;
}
