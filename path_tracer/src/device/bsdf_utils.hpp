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

inline __both__ float to_alpha(const float& roughness)
{
    return owl::max(alpha_min, sqr(roughness));
}

inline __both__ float schlick_fresnel(const float& cos_theta, const float& ior)
{
    auto const r0{sqr((1.0f - ior) / (1.0f + ior))};
    auto const m{owl::clamp(1.0f - cos_theta, 0.0f, 1.0f)};
    auto const m2{m * m};
    return r0 + (1.0f - r0) * (m2 * m2 * m);
}

inline __both__ float lambda(const float& tan_theta, const float& alpha)
{
    auto const abs_tan_theta_h{owl::abs(tan_theta)};
    if (isinf(abs_tan_theta_h)) return 0.0f;
    auto const alpha2_tan2_theta{sqr(alpha * abs_tan_theta_h)};
    return (-1.0f + owl::sqrt(1.0f + alpha2_tan2_theta)) / 2.0f;
}

inline __both__ float g_smith(const float& tan_theta, const float& alpha)
{
    return 1.0f / (1.0f + lambda(owl::abs(tan_theta), alpha));
}

inline __both__ float d_gtr1(const float& cos_theta, const float& alpha)
{
    if (alpha >= 1.0f) return inv_pi;
    auto const alpha2{sqr(alpha)};
    auto const t{1.0f + (alpha2 - 1.0f) * sqr(owl::abs(cos_theta))};
    return (alpha2 - 1.0f) / (pi * logf(alpha2) * t);
}

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

inline __both__ float d_gtr2(const float& cos_theta, const float& alpha)
{
    auto const alpha2{sqr(alpha)};
    auto const t{1.0f + (alpha2 - 1.0f) * sqr(cos_theta)};
    return alpha2 / (pi * sqr(t));
}

inline __both__ vec3 sample_gtr2_vndf(vec3 const& wo, const float& alpha, const vec2& u)
{
    auto const wh{normalize(vec3(alpha * wo.x, alpha * wo.y, wo.z))};
    auto const length_sqr = wh.x * wh.x + wh.y * wh.y;

    auto const T1{length_sqr > 0.0f ? vec3{-wh.y, wh.x, 0.0f} * (1.0f / owl::sqrt(length_sqr)) : vec3{1, 0, 0}};
    auto const T2 = owl::cross(wh, T1);

    auto const r = owl::sqrt(u.x);
    auto const phi = two_pi * u.y;
    auto const s = 0.5f * (1.0f + wh.z);

    auto const t1 = r * owl::cos(phi);
    auto const t2 = (1.0f - s) * owl::sqrt(1.0f - sqr(t1)) + s * r * owl::sin(phi);

    auto const nh{t1 * T1 + t2 * T2 + owl::sqrt(owl::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * wh};

    return owl::normalize(vec3{alpha * nh.x, alpha * nh.y, owl::max(0.0f, nh.z)});
}

inline __both__ float pdf_gtr2(vec3 const& wo, vec3 const& wh, const float& alpha)
{
    auto const dr{d_gtr2(cos_theta(wh), alpha)};
    auto const gr{g_smith(owl::abs(tan_theta(wo)), alpha)};
    return dr * gr * owl::abs(cos_theta(wh)) / (4.0f * owl::abs(cos_theta(wo)));
}

#endif // !PATH_TRACER_BSDF_UITLS_HPP
