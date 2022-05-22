#ifndef PATH_TRACER_DEVICE_HPP
#define PATH_TRACER_DEVICE_HPP

#include "device_global.hpp"
#include "random.hpp"

enum class scatter_event
{
    BOUNCED = 1 << 0,
    CANCELLED = 1 << 1,
    MISS = 1 << 2,
    NONE = 1 << 3
};

struct interface_data
{
    /* triangle points */
    vec3 TRI[3];

    /* hit position */
    vec3 P;

    /* shading normal */
    vec3 N;

    /* geometric normal */
    vec3 Ng;

    /* view direction (wo or V) */
    vec3 V;

    /* barycentrics */
    vec2 UV;

    /* thit */
    float t;

    /* primitive id => 0 if not exists */
    int32_t prim;

    /* material id for LP reference */
    int32_t matId;

    /* light id for LP reference */
    int32_t lightId;
};

struct per_ray_data
{
    random& random;
    scatter_event scatterEvent;
    interface_data* is;
    material_data* ms;
};

#endif // !PATH_TRACER_DEVICE_HPP
