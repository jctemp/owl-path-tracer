#ifndef PATH_TRACER_SAMPLE_METHODS_HPP
#define PATH_TRACER_SAMPLE_METHODS_HPP


#include "device.hpp"
#include "Math.hpp"

/*
* - http://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf.
* - https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
*/

__device__ vec2 sample_uniform_disk(const vec2& rand)
{
    float phi{ two_pi * rand.y };
    float r{ owl::sqrt(rand.x) };
    return { r * owl::cos(phi),r * owl::sin(phi) };
}

__device__ vec2 sample_concentric_disk(const vec2& rand)
{
    // re-scale rand to be between [-1,1]
    float dx{ 2.0f * rand.x - 1 };
    float dy{ 2.0f * rand.y - 1 };

    // handle degenerated origin
    if (dx == 0 && dy == 0)
        return vec2 {0.0f};

    // handle mapping unit square to unit disk
    float phi, r;
    if (std::abs(dx) > std::abs(dy))
    {
        r = dx;
        phi = pi_over_four * (dy / dx);
    }
    else
    {
        r = dy;
        phi = pi_over_two - pi_over_four * (dx / dy);
    }
    return { r * owl::cos(phi),r * owl::sin(phi) };
}

__device__ vec3 sample_uniform_sphere(const vec2& rand)
{
    float z{ 1.0f - 2.0f * rand.x };
    float r{ sqrtf(fmaxf(0.0f, 1.0f - z * z)) };
    float phi{ two_pi * rand.y };
    float x = r * owl::cos(phi);
    float y = r * owl::sin(phi);
    return vec3{ x, y, z };
}

__device__ vec3 sample_cosine_hemisphere(vec2 const& rand)
{
    // 1. sample unit circle and save position into rand.u, rand.v
    vec2 circle_points{sample_concentric_disk(rand)};
    // 2. calculate cosTheta => 1 = rand.u^2 + rand.v^2 => cos = 1 - (rand.u^2 + rand.v^2)
    float cos_theta{ owl::sqrt(owl::max(0.0f, 1.0f - sqr(circle_points.x) - sqr(circle_points.y))) };
    return vec3{ circle_points.x, circle_points.y, cos_theta };
}

__device__ float pdf_cosine_hemisphere(vec3 const& w_o, vec3 const& w_i)
{
    return absCosTheta(w_i) * inv_pi;
}

__device__ vec3 sample_uniform_hemisphere(vec2 const& rand)
{
    float z{ rand.x };
    float r{ sqrtf(fmaxf(0.0f, 1.0f - z * z)) };
    float phi = two_pi * rand.y;
    float x = r * owl::cos(phi);
    float y = r * owl::sin(phi);
    return vec3{ x, y, z };
}

__device__ float pdf_uniform_hemisphere(vec3 const& w_o, vec3 const& w_i)
{
    return 0.5f * inv_pi;
}


#endif //PATH_TRACER_SAMPLE_METHODS_HPP
