#ifndef DISNEY_BRDF_HPP
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

__device__ void sample_disney_diffuse(material_data const& m, vec3 const& wo, Random& rand, vec3& wi, vec3& f, float& pdf)
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

__device__ void sample_disney_subsurface(material_data const& m, vec3 const& wo, Random& rand, vec3& wi, vec3& f, float& pdf)
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


__device__ vec3 fDisneySheen(material_data const& mat, vec3 const& V, vec3 const& L)
{
    vec3 H{L + V};
    if (H.x == 0 && H.y == 0 && H.z == 0) return vec3{0.0f};
    if (mat.sheen <= 0.0f) return vec3{0.0f};

    H = normalize(H);
    float cosThetaD{dot(L, H)};

    vec3 tint{calculate_tint(mat.base_color)};
    return mat.sheen * lerp(vec3{1.0f}, tint, vec3{mat.sheen_tint}) * schlick_fresnel(cosThetaD, mat.ior);
}


__device__ float pdfDisneySheen(material_data const& mat, vec3 const& V, vec3 const& L)
{
    return pdf_cosine_hemisphere(V, L);
}


__device__ void sampleDisneySheen(material_data const& mat, vec3 const& V, vec3& L,
                                  Random& rand, vec3& bsdf, float& pdf)
{
    L = sample_cosine_hemisphere({rand.random(), rand.random()});
    pdf = pdfDisneySheen(mat, V, L);
    bsdf = fDisneySheen(mat, V, L) * owl::abs(cos_theta(L));
}

__device__ vec3 fDisneyClearcoat(material_data const& mat, vec3 const& V, vec3 const& L)
{
    vec3 H{L + V};
    if (H.x == 0 && H.y == 0 && H.z == 0) return vec3{0.0f};
    H = normalize(H);

    // Clearcoat has ior = 1.5 hardcoded -> F0 = 0.04. It then uses the
    // GTR1 distribution, which has even fatter tails than Trowbridge-Reitz
    // (which is GTR2). The geometric term always based on alpha = 0.25
    // - pbrt book v3
    float alphaG{(1 - mat.clearcoat_gloss) * 0.1f + mat.clearcoat_gloss * 0.001f};
    float Dr{d_gtr1(owl::abs(cos_theta(H)), alphaG)};
    float F{schlick_fresnel(owl::abs(cos_theta(H)), 1.5f)};
    float Fr{lerp(.04f, 1.0f, F)};
    float Gr{g_smith(owl::abs(cos_theta(V)), .25f) * g_smith(owl::abs(cos_theta(L)), .25f)};

    return mat.clearcoat * Gr * Fr * Dr / (4.0f * owl::abs(cos_theta(H)));
}


__device__ float pdfDisneyClearcoat(material_data const& mat, vec3 const& V, vec3 const& L)
{
    vec3 H{L + V};
    if (H.x == 0 && H.y == 0 && H.z == 0) return 0.0f;
    H = normalize(H);

    // The sampling routine samples H exactly from the GTR1 distribution.
    // Thus, the final value of the PDF is just the value of the
    // distribution for H converted to a measure with respect to the
    // surface normal.
    // - pbrt book v3
    float alphaG{(1 - mat.clearcoat_gloss) * 0.1f + mat.clearcoat_gloss * 0.001f};
    float Dr{d_gtr1(owl::abs(cos_theta(H)), alphaG)};
    return Dr * owl::abs(cos_theta(H)) / (4.0f * owl::abs(cos_theta(V)));
}


__device__ void sampleDisneyClearcoat(material_data const& mat, vec3 const& V, vec3& L,
                                      Random& rand, vec3& bsdf, float& pdf)
// there is no visible normal sampling for BRDF because the Clearcoat has no
// physical meaning
// Apply importance sampling to D * dot(N * H) / (4 * dot(H, V)) by choosing normal
// proportional to D and reflect it at H
{
    float alphaG{(1 - mat.clearcoat_gloss) * 0.1f + mat.clearcoat_gloss * 0.001f};
    float alpha2{alphaG * alphaG};
    float cosTheta{sqrtf(fmax(0.0f, (1 - powf(alpha2, 1 - rand.random())) / (1 - alpha2)))};
    float sinTheta{sqrtf(fmax(0.0f, 1 - cosTheta * cosTheta))};
    float phi{two_pi * rand.random()};

    vec3 H{to_sphere_coordinates(sinTheta, cosTheta, phi)};

    if (!same_hemisphere(V, H)) H = -H;

    bsdf = vec3{0.0f};
    pdf = 0.0f;

    L = reflect(V, H);
    if (!same_hemisphere(V, L)) return;

    pdf = pdfDisneyClearcoat(mat, V, L);
    bsdf = fDisneyClearcoat(mat, V, L);
}

__device__ vec3 fDisneyMicrofacet(material_data const& mat, vec3 const& V, vec3 const& L)
{
    if (!same_hemisphere(V, L)) return vec3{0.0f};
    float NdotV{owl::abs(cos_theta(V))};
    if (NdotV == 0) return vec3{0.0f};

    vec3 H{normalize(V + L)};

    float alpha{to_alpha(mat.roughness)};
    float Dr{d_gtr2(cos_theta(H), alpha)};
    vec3 Fr{lerp(mat.base_color, 1.0f - mat.base_color, {schlick_fresnel(dot(H, V), mat.ior)})};
    float Gr{g_smith(abs(tan_theta(V)), alpha) *
             g_smith(abs(tan_theta(L)), alpha)};

    return Dr * Fr * Gr / (4.0f * owl::abs(cos_theta(L)));
}


__device__ float pdfDisneyMicrofacet(material_data const& mat, vec3 const& V, vec3 const& L)
{
    if (!same_hemisphere(V, L)) return 0.0f;

    vec3 H{L + V};
    if (H.x == 0 && H.y == 0 && H.z == 0) return 0.0f;
    H = normalize(H);

    float alpha{to_alpha(mat.roughness)};
    return pdf_gtr2(V, H, alpha);
}


__device__ void sampleDisneyMicrofacet(material_data const& mat, vec3 const& V, vec3& L,
                                       Random& rand, vec3& bsdf, float& pdf)
{
    float alpha{to_alpha(mat.roughness)};
    vec3 H{sample_gtr2_vndf(V, alpha, {rand.random(), rand.random()})};

    L = reflect(V, H);

    bsdf = vec3{0.0f};
    pdf = 0.0f;

    if (!same_hemisphere(V, L)) return;

    bsdf = fDisneyMicrofacet(mat, V, L);
    pdf = pdfDisneyMicrofacet(mat, V, L);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           disney'S PRINCIPLED BSDF
//
//  Specular BSDF ───────┐ Metallic BRDF ┐
//                       │               │
//  Dielectric BRDF ─── lerp ─────────── lerp ─── disney BSDF
//  + Subsurface
//

__device__ void sampleDisneyBSDF(material_data const& mat, vec3 const& V, vec3& L,
                                 Random& rand, vec3& bsdf, float& pdf)
{
    bsdf = vec3{0.0f};
    pdf = 0.0f;

    float r1 = rand.random(), r2 = rand.random();

    float specularWeight = mat.metallic;
    float diffuseWeight = 1 - mat.metallic;
    float clearcoatWeight = 1.0f * o_saturate(mat.clearcoat);

    float norm = 1.0f / (clearcoatWeight + diffuseWeight + specularWeight);

    float pSpecular = specularWeight * norm;
    float pDiffuse = diffuseWeight * norm;
    float pClearcoat = clearcoatWeight * norm;

    float cdf[3]{};
    cdf[0] = pDiffuse;
    cdf[1] = pSpecular + cdf[0];
    cdf[2] = pClearcoat + cdf[1];


    // diffuse
    if (r1 < cdf[0])
    {
        vec3 H{normalize(L + V)};

        if (H.z < 0.0f) H = -H;

        L = sample_cosine_hemisphere({rand.random(), rand.random()});
        pdf = pdf_cosine_hemisphere(V, L);

        auto diffuse = f_disney_diffuse(mat, V, L);
        auto ss = f_disney_subsurface(mat, V, L);
        auto retro = f_disney_retro(mat, V, L);
        auto sheen = fDisneySheen(mat, V, L);

        auto diffuseWeight = 1 - mat.subsurface;
        auto ssWeight = mat.subsurface;

        bsdf += (diffuse + retro) * diffuseWeight;
        bsdf += ss * ssWeight;
        bsdf += sheen;
        bsdf *= owl::abs(cos_theta(L));
    }

        // specular reflective
    else if (r1 < cdf[1])
    {
        float alpha{to_alpha(mat.roughness)};
        vec3 H{sample_gtr2_vndf(V, alpha, {rand.random(), rand.random()})};

        L = normalize(reflect(V, H));

        if (!same_hemisphere(V, L)) return;

        bsdf = fDisneyMicrofacet(mat, V, L);
        pdf = pdfDisneyMicrofacet(mat, V, L);
    }

        // clearcoat
    else
    {
        sampleDisneyClearcoat(mat, V, L, rand, bsdf, pdf);
    }

}

#endif // !DISNEY_BRDF_HPP
