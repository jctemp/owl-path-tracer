#ifndef PATH_TRACER_DEVICE_HPP
#define PATH_TRACER_DEVICE_HPP

#include "device_global.hpp"
#include "random.hpp"

enum class scatter_event
{
    hit = 1 << 0,
    miss = 1 << 1,
    none = 1 << 2
};

struct hit_data
{
    int32_t primitive_index;
    int32_t material_index;
    int32_t mesh_index;
    int32_t light_index;
    vec2 barycentric;
    vec3 wo;
    float t;
};

struct per_ray_data
{
    random& random;
    scatter_event scatter_event;
    hit_data* hd;
    material_data* ms;
};

#endif // !PATH_TRACER_DEVICE_HPP
