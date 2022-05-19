#ifndef PATH_TRACER_CAMERA_HPP
#define PATH_TRACER_CAMERA_HPP

#include "types.hpp"

struct camera
{
    vec3 const look_from;
    vec3 const look_at;
    vec3 const look_up;
    float const vertical_fov;
};

struct camera_data
{
    vec3 origin;
    vec3 llc;
    vec3 horizontal;
    vec3 vertical;
};

extern camera_data to_camera_data(camera const& c, ivec2 const& buffer_size);

#endif // !PATH_TRACER_CAMERA_HPP
