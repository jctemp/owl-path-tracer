#ifndef PATH_TRACER_BSDF_UITLS_HPP
#define PATH_TRACER_BSDF_UITLS_HPP

#include "sample_methods.hpp"

// https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf
// https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.2297&rep=rep1&type=pdf
// https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
// https://jcgt.org/published/0007/04/01/

inline __both__ float luminance(const vec3& color)
{
    return owl::dot(vec3{0.2126f, 0.7152f, 0.0722f}, color);
}

inline __both__ vec3 calculate_tint(const vec3& base_color)
{
    auto const lum{luminance(base_color)};
    return (lum > 0.0f) ? base_color * (1.0f / lum) : vec3{1.0f};
}

// Physically Based Shading at Disney 2012 - Section 5.1 page 15
inline __both__ vec2 to_alpha(float roughness, float anisotopic = 0.0f)
{
    auto const aspect{owl::sqrt(1.0f - 0.9f * anisotopic)};
    return vec2{owl::max(alpha_min, sqr(roughness) / aspect), owl::max(alpha_min, sqr(roughness) * aspect)};
}

inline __both__ float f_schlick(float cos_theta, float ior)
{
    auto const r0{sqr((1.0f - ior) / (1.0f + ior))};
    auto const m{owl::clamp(1.0f - cos_theta, 0.0f, 1.0f)};
    auto const m2{m * m};
    return r0 + (1.0f - r0) * (m2 * m2 * m);
}

// An Inexpensive BRDF Model for Physically Based Rendering - Schlick - equ. (15)
inline __both__ float f_schlick(float u)
{
    auto const m{owl::clamp(1.0f - u, 0.0f, 1.0f)};
    auto const m2{m * m};
    return m2 * m2 * m;
}

// Physically Based Shading at Disney 2012 - B.2 GTR equ. (4)
inline __both__ float d_gtr1(vec3 const& wh, float alpha)
{
    if (alpha >= 1.0f) return inv_pi;
    auto const alpha2{sqr(alpha)};
    auto const t{1.0f + (alpha2 - 1.0f) * sqr(cos_theta(wh))};
    return (alpha2 - 1.0f) / (pi * logf(alpha2) * t);
}

// Physically Based Shading at Disney 2012 - B, GTR equ. (2) and (9)
inline __both__ vec3 sample_gtr1(vec3 const& wo, const float& alpha_g, const vec2& u)
{
    auto const alpha2{sqr(alpha_g)};
    auto const cos_theta{owl::sqrt(owl::max(0.0f, (1.0f - powf(alpha2, 1.0f - u[0])) / (1.0f - alpha2)))};
    auto const sin_theta{owl::sqrt(owl::max(0.0f, 1.0f - sqr(cos_theta)))};
    auto const phi{two_pi * u[1]};

    auto wh{to_sphere_coordinates(sin_theta, cos_theta, phi)};
    if (!same_hemisphere(wo, wh)) wh = -wh;
    return wh;
}

// Understanding Masking-Shadowing Functions for Microfacet-Based BRDFs (85)
inline __both__ float d_gtr2(vec3 const& wh, float ax, float ay)
{
    auto const tan2_theta {sqr(tan_theta(wh))};
    if (isinf(tan2_theta)) return 0.0f;
    auto const cos4_theta {sqr(cos_theta(wh)) * sqr(cos_theta(wh))};
    auto const e{1.0f + tan2_theta * (sqr(cos_phi(wh) / ax) + sqr(sin_phi(wh) / ay))};
    return 1.0f / (pi * ax * ay * cos4_theta * sqr(e));
}

// Understanding Masking-Shadowing Functions for Microfacet-Based BRDFs (Heitz, 2017) equ. (72)
inline __both__ float g_smith_lambda(vec3 const& w, float ax, float ay)
{
    auto const abs_tan_theta {owl::abs(tan_theta(w))};
    if (isinf(abs_tan_theta)) return 0.0f;
    auto const a0{owl::sqrt(sqr(ax * cos_phi(w)) + sqr(ay * sin_phi(w)))};
    return (-1.0f + owl::sqrt(1.0f + sqr(a0) * sqr(abs_tan_theta))) / 2.0f;
}

// Understanding Masking-Shadowing Functions for Microfacet-Based BRDFs (Heitz, 2017) equ. (43)
inline __both__ float g1_smith(vec3 const& w, float ax, float ay)
{
    return 1.0f / (1.0f + g_smith_lambda(w, ax, ay));
}

// Understanding Masking-Shadowing Functions for Microfacet-Based BRDFs (Heitz, 2017) equ. (99)
inline __both__ float g2_smith_correlated(vec3 const& wo, vec3 const& wi, vec3 const& wh, float ax, float ay)
{
    auto const lambda_o{g_smith_lambda(wo, ax, ay)};
    auto const lambda_i{g_smith_lambda(wi, ax, ay)};
    return 1.0f / (1.0f + lambda_o + lambda_i);
}

// Sampling the GGX Distribution of visible normals - listing 1. complete implementation
inline __both__ vec3 sample_gtr2_vndf(vec3 const& wo, float ax, float ay, const vec2& u)
{
    auto const wh{normalize(vec3(ax * wo.x, ay * wo.y, wo.z))};
    auto const length_sqr = wh.x * wh.x + wh.y * wh.y;

    auto const T1{length_sqr > 0.0f ? vec3{-wh.y, wh.x, 0.0f} * (1.0f / owl::sqrt(length_sqr)) : vec3{1, 0, 0}};
    auto const T2 = owl::cross(wh, T1);

    float r = sqrtf(u.x);
    float phi = 2.0f * pi * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.0f + wh.z);
    t2 = (1.0f - s) * sqrtf(1.0f - t1*t1) + s*t2;

    auto const nh{t1 * T1 + t2 * T2 + owl::sqrt(owl::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * wh};

    return owl::normalize(vec3{ax * nh.x, ay * nh.y, owl::max(0.0f, nh.z)});
}

#endif // !PATH_TRACER_BSDF_UITLS_HPP
