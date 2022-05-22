#ifndef PATH_TRACER_DEVICE_HPP
#define PATH_TRACER_DEVICE_HPP

#include "device_global.hpp"
#include "random.hpp"

enum class scatter_event
{
    bounced = 1 << 0,
    missed = 1 << 1,
    none = 1 << 2
};

struct interface_data
{
    /* triangle points */
    vec3 triangle_points[3];

    /* hit position */
    vec3 position;

    /* shading normal */
    vec3 normal;

    /* geometric normal */
    vec3 normal_geometric;

    /* view direction (wo or V) */
    vec3 wo;

    /* barycentrics */
    vec2 uv;

    /* thit */
    float t;

    /* primitive id => 0 if not exists */
    int32_t prim;

    /* material id for LP reference */
    int32_t material_id;

    /* light id for LP reference */
    int32_t light_id;
};

struct per_ray_data
{
    random& random;
    scatter_event scatter_event;
    interface_data* is;
    material_data* ms;
};

#endif // !PATH_TRACER_DEVICE_HPP
