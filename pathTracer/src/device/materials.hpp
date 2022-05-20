﻿#ifndef DISNEY_BRDF_HPP
#define DISNEY_BRDF_HPP
#pragma once

#include "sample_methods.hpp"
#include "device/bsdf_utils.hpp"
#include "types.hpp"
#include "macros.hpp"

// https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf

__device__ vec3 f_lambert(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    return m.base_color * inv_pi;
}

__device__ float pdf_lambert(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    return pdf_cosine_hemisphere(wo, wi);
}

__device__ void sample_lambert(material_data const& m, vec3 const& wo, Random& rand, vec3& wi, vec3& f, float& pdf)
{
    sample_cosine_hemisphere(rand.random<vec2>());
    pdf = pdf_lambert(m, wo, wi);
    f = f_lambert(m, wo, wi);
}


__device__ vec3 f_disney_diffuse(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    // Burley 2015, eq (4).
    auto const n_dot_wo{owl::abs(cos_theta(wo))};
    auto const n_dot_wi{owl::abs(cos_theta(wi))};

    auto const fresnel_wo{schlick_fresnel(n_dot_wo, m.ior)};
    auto const fresnel_wi{schlick_fresnel(n_dot_wi, m.ior)};

    return m.base_color * inv_pi * (1.0f - fresnel_wi / 2.0f) * (1.0f - fresnel_wo / 2.0f);
}

__device__ float pdf_disney_diffuse(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    return pdf_cosine_hemisphere(wo, wi);
}

__device__ void
sample_disney_diffuse(material_data const& m, vec3 const& wo, Random& rand, vec3& wi, vec3& f, float& pdf)
{
    wi = sample_cosine_hemisphere({rand.random(), rand.random()});
    pdf = pdf_disney_diffuse(m, wo, wi);
    f = f_disney_diffuse(m, wo, wi) * owl::abs(cos_theta(wi));
}


__device__ vec3 f_disney_subsurface(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    auto wh{wi + wo};
    if (all_zero(wh)) return vec3{0.0f};
    wh = owl::normalize(wh);

    // Hanrahan - Krueger BRDF approximation of the BSSRDF
    // fss90 used to "flatten" retro-reflection based on roughness
    // 1.25 scale is used to (roughly) preserve albedo

    auto const cos_theta_d{owl::dot(wi, wh)};
    auto const n_dot_wo{owl::abs(cos_theta(wo))};
    auto const n_dot_wi{owl::abs(cos_theta(wi))};
    auto const fresnel_wo{schlick_fresnel(n_dot_wo, m.ior)};
    auto const fresnel_wi{schlick_fresnel(n_dot_wi, m.ior)};
    auto const fss90{sqr(cos_theta_d) * m.roughness};
    auto const fss{lerp(1.0f, fss90, fresnel_wo) * lerp(1.0f, fss90, fresnel_wi)};
    auto const fs{1.25f * (fss * (1.0f / (n_dot_wo + n_dot_wi) - 0.5f) + 0.5f)};

    return m.subsurface_color * inv_pi * fs;
}

__device__ float pdf_disney_subsurface(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    return pdf_cosine_hemisphere(wo, wi);
}

__device__ void
sample_disney_subsurface(material_data const& m, vec3 const& wo, Random& rand, vec3& wi, vec3& f, float& pdf)
{
    wi = sample_cosine_hemisphere({rand.random(), rand.random()});
    pdf = pdf_disney_subsurface(m, wo, wi);
    f = f_disney_subsurface(m, wo, wi) * owl::abs(cos_theta(wi));
}


__device__ vec3 f_disney_retro(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    auto wh{wi + wo};
    if (all_zero(wh)) return vec3{0.0f};
    wh = owl::normalize(wh);

    // Burley 2015, eq (4).

    auto const cos_theta_d{owl::dot(wi, wh)};
    auto const n_dot_wo{owl::abs(cos_theta(wo))};
    auto const n_dot_wi{owl::abs(cos_theta(wi))};
    auto const fresnel_wo{schlick_fresnel(n_dot_wo, m.ior)};
    auto const fresnel_wi{schlick_fresnel(n_dot_wi, m.ior)};
    auto const rr{2 * m.roughness * sqr(cos_theta_d)};

    return m.base_color * inv_pi * rr * (fresnel_wo + fresnel_wi + fresnel_wo * fresnel_wi * (rr - 1));
}

__device__ float pdf_disney_retro(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    return pdf_cosine_hemisphere(wo, wi);
}

__device__ void sample_disney_retro(material_data const& m, vec3 const& wo, Random& rand, vec3& wi, vec3& f, float& pdf)
{
    wi = sample_cosine_hemisphere(rand.random<vec2>());
    pdf = pdf_disney_retro(m, wo, wi);
    f = f_disney_retro(m, wo, wi) * owl::abs(cos_theta(wi));
}


__device__ vec3 f_disney_sheen(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    if (m.sheen <= 0.0f) return vec3{0.0f};

    auto wh{wi + wo};
    if (all_zero(wh)) return vec3{0.0f};
    wh = owl::normalize(wh);

    auto const cos_theta_d{owl::dot(wi, wh)};
    auto const tint{calculate_tint(m.base_color)};

    return m.sheen * lerp(vec3{1.0f}, tint, vec3{m.sheen_tint}) * schlick_fresnel(cos_theta_d, m.ior);
}

__device__ float pdf_disney_sheen(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    return pdf_cosine_hemisphere(wo, wi);
}

__device__ void sample_disney_sheen(material_data const& m, vec3 const& wo, Random& rand, vec3& wi, vec3& f, float& pdf)
{
    wi = sample_cosine_hemisphere({rand.random(), rand.random()});
    pdf = pdf_disney_sheen(m, wo, wi);
    f = f_disney_sheen(m, wo, wi) * owl::abs(cos_theta(wi));
}


__device__ vec3 f_disney_clearcoat(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    auto wh{wi + wo};
    if (all_zero(wh)) return vec3{0.0f};
    wh = owl::normalize(wh);

    // Clearcoat has ior = 1.5 hardcoded -> F0 = 0.04. It then uses the
    // GTR1 distribution, which has even fatter tails than Trowbridge-Reitz
    // (which is GTR2). The geometric term always based on alpha = 0.25

    auto const alpha_g{(1 - m.clearcoat_gloss) * 0.1f + m.clearcoat_gloss * 0.001f};
    auto const dr{d_gtr1(owl::abs(cos_theta(wh)), alpha_g)};
    auto const fr{lerp(.04f, 1.0f, schlick_fresnel(owl::abs(cos_theta(wh)), 1.5f))};
    auto const gr{g_smith(owl::abs(tan_theta(wo)), .25f) *
                  g_smith(owl::abs(tan_theta(wi)), .25f)};

    return m.clearcoat * gr * fr * dr / (4.0f * owl::abs(cos_theta(wh)));
}

__device__ float pdf_disney_clearcoat(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    auto wh{wi + wo};
    if (all_zero(wh)) return 0.0f;
    wh = owl::normalize(wh);

    // The sampling routine samples wh exactly from the GTR1 distribution.
    // Thus, the final value of the PDF is just the value of the
    // distribution for wh converted to a measure with respect to the
    // surface normal.

    auto const alpha_g{(1 - m.clearcoat_gloss) * 0.1f + m.clearcoat_gloss * 0.001f};
    auto const dr{d_gtr1(owl::abs(cos_theta(wh)), alpha_g)};
    return dr * owl::abs(cos_theta(wh)) / (4.0f * owl::abs(cos_theta(wo)));
}

__device__ void
sample_disney_clearcoat(material_data const& m, vec3 const& wo, Random& rand, vec3& wi, vec3& f, float& pdf)
{
    // there is no visible normal sampling for BRDF because the Clearcoat has no
    // physical meaning
    // Apply importance sampling to D * dot(N * H) / (4 * dot(H, V)) by choosing normal
    // proportional to D and reflect it at H

    auto const alpha_g{(1 - m.clearcoat_gloss) * 0.1f + m.clearcoat_gloss * 0.001f};
    auto const alpha2{sqr(alpha_g)};
    auto const cos_theta{owl::sqrt(owl::max(0.0f, (1 - powf(alpha2, 1 - rand())) / (1 - alpha2)))};
    auto const sin_theta{owl::sqrt(owl::max(0.0f, 1 - sqr(cos_theta)))};
    auto const phi{two_pi * rand()};

    auto wh{to_sphere_coordinates(sin_theta, cos_theta, phi)};
    if (!same_hemisphere(wo, wh)) wh = -wh;

    f = vec3{0.0f};
    pdf = 0.0f;

    wi = reflect(wo, wh);
    if (!same_hemisphere(wo, wi)) return;

    pdf = pdf_disney_clearcoat(m, wo, wi);
    f = f_disney_clearcoat(m, wo, wi);
}


__device__ vec3 f_disney_microfacet(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    if (owl::abs(cos_theta(wo)) == 0) return vec3{0.0f};

    auto wh{wi + wo};
    if (all_zero(wh)) return vec3{0.0f};
    wh = owl::normalize(wh);

    auto const alpha{to_alpha(m.roughness)};
    auto const dr{d_gtr2(cos_theta(wh), alpha)};
    auto const fr{lerp(m.base_color, 1.0f - m.base_color, schlick_fresnel(dot(wh, wo), m.ior))};
    auto const gr{g_smith(abs(tan_theta(wo)), alpha) *
                  g_smith(abs(tan_theta(wi)), alpha)};

    return dr * fr * gr / (4.0f * owl::abs(cos_theta(wi)));
}

__device__ float pdf_disney_microfacet(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    auto wh{wi + wo};
    if (all_zero(wh)) return 0.0f;
    wh = owl::normalize(wh);

    auto const alpha{to_alpha(m.roughness)};
    return pdf_gtr2(wo, wh, alpha);
}

__device__ void
sample_disney_microfacet(material_data const& m, vec3 const& wo, Random& rand, vec3& wi, vec3& f, float& pdf)
{
    auto const alpha{to_alpha(m.roughness)};
    auto const wh{sample_gtr2_vndf(wo, alpha, rand.random<vec2>())};

    wi = reflect(wo, wh);

    f = vec3{0.0f};
    pdf = 0.0f;

    if (!same_hemisphere(wo, wi)) return;

    f = f_disney_microfacet(m, wo, wi);
    pdf = pdf_disney_microfacet(m, wo, wi);
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           disney'S PRINCIPLED BSDF
//
//  Specular BSDF ───────┐ Metallic BRDF ┐
//                       │               │
//  Dielectric BRDF ─── lerp ─────────── lerp ─── disney BSDF
//  + Subsurface
//

__device__ void
sample_disney_bsdf(material_data const& mat, vec3 const& wo, Random& rand, vec3& wi, vec3& f, float& pdf)
{
    f = vec3{0.0f};
    pdf = 0.0f;

    auto const r1 = rand();
    auto const specular_weight  {mat.metallic};
    auto const diffuse_weight   {1 - mat.metallic};
    auto const clearcoat_weight {1.0f * o_saturate(mat.clearcoat)};
    auto const weighted_sum     {1.0f / (clearcoat_weight + diffuse_weight + specular_weight)};

    auto const p_specular  {specular_weight * weighted_sum};
    auto const p_diffuse   {diffuse_weight * weighted_sum};
    auto const p_clearcoat {clearcoat_weight * weighted_sum};

    float cdf[3]{};
    cdf[0] = p_diffuse;
    cdf[1] = p_specular + cdf[0];
    cdf[2] = p_clearcoat + cdf[1];

    // diffuse
    if (r1 < cdf[0])
    {
        wi = sample_cosine_hemisphere(rand.random<vec2>());
        pdf = pdf_cosine_hemisphere(wo, wi);

        auto const diffuse = f_disney_diffuse(mat, wo, wi);
        auto const ss = f_disney_subsurface(mat, wo, wi);
        auto const retro = f_disney_retro(mat, wo, wi);
        auto const sheen = f_disney_sheen(mat, wo, wi);

        auto const fd_weight = 1 - mat.subsurface;
        auto const fs_weight = mat.subsurface;

        f += (diffuse + retro) * fd_weight;
        f += ss * fs_weight;
        f += sheen;
        f *= owl::abs(cos_theta(wi));
    }

        // specular reflective
    else if (r1 < cdf[1])
    {
        auto const alpha{to_alpha(mat.roughness)};
        auto const wh{sample_gtr2_vndf(wo, alpha, rand.random<vec2>())};

        wi = normalize(reflect(wo, wh));
        if (!same_hemisphere(wo, wi)) return;

        f = f_disney_microfacet(mat, wo, wi);
        pdf = pdf_disney_microfacet(mat, wo, wi);
    }

        // clearcoat
    else
    {
        sample_disney_clearcoat(mat, wo, rand, wi, f, pdf);
    }

}

#endif // !DISNEY_BRDF_HPP
