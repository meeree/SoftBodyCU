#pragma once
#include <stddef.h>
#include <cstdio>
#include <cstdint>

namespace Graphics
{
    //! Sets up the graphics.
    int setup(float*& voltages, float const* positions, float const* normals, size_t N, size_t render_N);

    bool render(float const* voltages, void const* slice, int const slice_width, int const slice_height, bool& pause);
    void update_positions(float const* positions);
    void terminate_graphics();

    //! Sets size of maximum raw video size. If we exceed this, the raw video is converted to an mp4 and overwritten. 
    void set_max_raw_video_size(int const size_in_gb);

    //! Writes a frame to raw video file. If file size starts to exceed limit, it is converted to an mp4 and will be overwritten.
    void write_frame(FILE* out, uint8_t* data);

    //! replay utilities. Allows user to record a replay of voltages and replay it. 
    void setup_replay(int frame_cache_count); // Allocate frame_cache_count frames to store replay. 
    //! replay utilities. Allows user to record a replay of voltages and replay it. 
    void record_for_replay(float const* voltages);

    //! replay utilities. Allows user to record a replay of voltages and replay it. 
    void set_window_title(const char* title);

    //! helper function
    int get_width ();
    //! helper function
    int get_height ();

    //! helper function
    int get_slice_index();

    //! Add points to be rendered.
    void add_points(float const* points, int const N);

    //! Add lines to be rendered. N is number of lines, not vertices.
    void add_lines(int const* lines, int const N);

    //! Add triangles to be rendered. N is number of triangles, not vertices.
    void add_triangles(int const* triangles, int const N);

    typedef bool (*LinePredicate)(float const*, float const*);
    //! Load a connectivity as a collection of lines to be rendered.
    void load_in_connectivity(
        int const* connectivity_inds, int const* offset_into_inds,
        int const N, int const N_pts_to_use,
        LinePredicate pred = nullptr);

    //! Set color divisor, offset, and cutoff (the latter being in the range 0 to 1).
    //! Color is computed as t = (voltage - offset) / divisor where if t < cutoff it is not shown.
    void set_color_params(float divisor, float offset, float cutoff);

    float get_color_divisor();
    float get_color_offset();
    float get_color_cutoff();
};