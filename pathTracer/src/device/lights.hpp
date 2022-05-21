
#ifndef PATH_TRACER_LIGHTS_HPP
#define PATH_TRACER_LIGHTS_HPP

#include "types.hpp"
#include "macros.hpp"

__device__ float power_heuristic(float n_f, float pdf_f, float n_g, float pdf_g)
{
    float f = n_f * pdf_f;
    float g = n_g * pdf_g;
    return (f * f) / (f * f + g * g);
}

#endif //PATH_TRACER_LIGHTS_HPP
