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
inline __both__ float to_alpha(float roughness)
{
    return owl::max(alpha_min, sqr(roughness));
}

// Physically Based Shading at Disney 2012 - Addenda
inline __both__ vec2 to_alpha(float roughness, float anisotropy)
{
    auto const aspect{owl::sqrt(1.0f - 0.9f * anisotropy)};
    return vec2{to_alpha(roughness) / aspect, to_alpha(roughness) * aspect};
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

// Microfacet Models for Refraction through Rough Surfaces equ. (33)
inline __both__ float d_gtr2(vec3 const& wh, vec2 const& a)
{
    auto const tan2_theta {sqr(tan_theta(wh))};
    if (isinf(tan2_theta)) return 0.0f;
    auto const cos4_theta {sqr(cos_theta(wh)) * sqr(cos_theta(wh))};
    auto const e{sqr(cos_phi(wh) * a.x) + sqr(sin_phi(wh) * a.y) * tan2_theta};
    return 1.0f / (pi * a.x * a.y * cos4_theta * (1 + e) * (1 + e));
}

// Understanding Masking-Shadowing Functions for Microfacet-Based BRDFs (Heitz, 2017) equ. (72)
inline __both__ float lambda_gtr(vec3 const& w, vec2 const& a)
{
    auto const abs_tan_theta{owl::abs(tan_theta(w))};
    if (isinf(abs_tan_theta)) return 0.0f;
    auto const alpha{sqr(cos_phi(w) * a.x) + sqr(sin_phi(w) * a.y)};
    auto const alpha2_tan2_theta{sqr(alpha * abs_tan_theta)};
    return (-1.0f + owl::sqrt(1.0f + alpha2_tan2_theta)) / 2.0f;

}

// Microfacet Models for Refraction through Rough Surfaces equ. (51)
inline __both__ float g1_smith(vec3 const& w, float const& alpha)
{
    return 1.0f / (1.0f + lambda_gtr(w, alpha));
}

// Understanding Masking-Shadowing Functions for Microfacet-Based BRDFs (Heitz, 2017) equ. (99)
inline __both__ float g2_smith_correlated(vec3 const& wo, vec3 const& wi, vec3 const& wh, vec2 const& alpha)
{
    auto const lambda_o{lambda_gtr(wo, alpha)};
    auto const lambda_i{lambda_gtr(wi, alpha)};
    return 1.0f / (1.0f + lambda_o + lambda_i);
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

// Sampling the GGX Distribution of visible normals - listing 1. complete implementation
inline __both__ vec3 sample_gtr2_vndf(vec3 const& wo, const vec2& alpha, const vec2& u)
{
    auto const wh{normalize(vec3(alpha.x * wo.x, alpha.y * wo.y, wo.z))};
    auto const length_sqr = wh.x * wh.x + wh.y * wh.y;

    auto const T1{length_sqr > 0.0f ? vec3{-wh.y, wh.x, 0.0f} * (1.0f / owl::sqrt(length_sqr)) : vec3{1, 0, 0}};
    auto const T2 = owl::cross(wh, T1);

    auto const r = owl::sqrt(u.x);
    auto const phi = two_pi * u.y;
    auto const s = 0.5f * (1.0f + wh.z);

    auto const t1 = r * owl::cos(phi);
    auto const t2 = (1.0f - s) * owl::sqrt(1.0f - sqr(t1)) + s * r * owl::sin(phi);

    auto const nh{t1 * T1 + t2 * T2 + owl::sqrt(owl::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * wh};

    return owl::normalize(vec3{alpha.x * nh.x, alpha.y * nh.y, owl::max(0.0f, nh.z)});
}

#endif // !PATH_TRACER_BSDF_UITLS_HPP
