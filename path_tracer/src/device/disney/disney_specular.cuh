#ifndef PATH_TRACER_DISNEY_SPECULAR_CUH
#define PATH_TRACER_DISNEY_SPECULAR_CUH

#include "sample_methods.hpp"
#include "types.hpp"
#include "macros.hpp"
#include "disney_helper.cuh"

// References:
// - Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs [0]
// - Average irregularity representation of a rough surface for ray reflection [1]
// - Sampling the GGX Distribution of Visible Normals [2]
// - Physically Based Shading at Disney [3]
// - Microfacet Models for Refraction through Rough Surfaces [4]

/// eq. (86) from [0]
__both__ float lambda(vec3 const& w, float ax, float ay)
{
    auto const abs_tan_theta{tan_theta(w)};
    if (isinf(abs_tan_theta)) return 0.0f;

    // eq. (80) from [0]
    auto const alpha0{sqrt(sqr(cos_phi(w) * ax) + sqr(sin_phi(w) * ay))};
    auto const a{1.0f / (alpha0 * abs_tan_theta)};
    return (-1.0f + sqrt(1.0f + 1.0f / sqr(a))) / 2.0f;
}

/// eq. (43) from [0]
/// generalized version of the geometric term
__both__ float g1_smith(vec3 const& w, float ax, float ay)
{
    return 1.0f / (1.0f + lambda(w, ax, ay));
}

/// eq. (98) from [0]
/// this geometric term is overestimating
__both__ float g2_smith_separable(vec3 const& wo, vec3 const& wi, float ax, float ay)
{
    return g1_smith(wo, ax, ay) * g1_smith(wi, ax, ay);
}

/// eq. (99) from [0]
/// better solution because it is taking facet heights into account
/// can be improved by using the eq. (101) from [0]
__both__ float g2_smith_correlated(vec3 const& wo, vec3 const& wi, float ax, float ay)
{
    return 1.0f / (1.0f + lambda(wo, ax, ay) + lambda(wi, ax, ay));
}

/// Because of using the revised version of the smith g term one adopts the
/// eq. (85) from [0]
/// wm here is the normal vector of the microfacet [1]
__both__ float d_gtr_2(vec3 const& wm, float ax, float ay)
{
    auto const tan2_theta{sqr(tan_theta(wm))};
    if (isinf(tan2_theta)) return 0.0f;
    auto const cos4_theta{sqr(sqr(cos_theta(wm)))};
    auto const e{1.0f + tan2_theta * (sqr(cos_phi(wm)) / sqr(ax) + sqr(sin_phi(wm)) / sqr(ay))};
    return 1.0f / (pi * ax * ay * cos4_theta * sqr(e));
}

/// https://agraphicsguy.wordpress.com/2018/07/18/sampling-anisotropic-microfacet-brdf/
__both__ vec3 sample_gtr2_ndf(vec3 const& wo, float ax, float ay, const vec2& u)
{
    float offset[5] = {0.0f, 1.0f, 1.0f, 2.0f, 2.0f};
    auto const i = u.v == 0.25f ? 0 : (int) (u.v * 4.0f);
    auto const phi = atan((ay / ax) * tan(two_pi * u.v)) + offset[i] * pi;
    auto const sin_phi = sin(phi);
    auto const sin_phi_sq = sin_phi * sin_phi;
    auto const cos_phi_sq = 1.0f - sin_phi_sq;
    auto const beta = 1.0f / (cos_phi_sq / sqr(ax) + sin_phi_sq / sqr(ay));
    auto const theta = atan(sqrt(beta * u.u / (1.0f - u.u)));
    return to_sphere_coordinates(theta, phi);
}

/// Sampling the Trowbridge-Reitz distribution of visible normals [2]
/// it allows for better sampling only relevant normals for the microfacet
__both__ vec3 sample_gtr2_vndf(vec3 const& wo, float ax, float ay, vec2 const& u)
{
    // transforming the view direction to the hemisphere configuration
    auto const N{normalize(vec3(ax * wo.x, ay * wo.y, wo.z))};

    // orthonormal basis (with special case if cross product is zero)
    // => basis (N, T, B)
    auto const length_sqr = sqr(N.x) + sqr(N.y);
    auto const T{length_sqr > 0.0f ? vec3{-N.y, N.x, 0.0f} *
                                     (1.0f / sqrt(length_sqr)) : vec3{1, 0, 0}};
    auto const B = cross(N, T);

    // parameterization of the projected area
    float r = sqrt(u.x);
    float phi = two_pi * u.y;
    float t = r * cos(phi);
    float b = r * sin(phi);
    float s = 0.5f * (1.0f + N.z);
    b = (1.0f - s) * sqrt(1.0f - sqr(t)) + s * b;

    // re-projection onto hemisphere
    auto const nh{t * T + b * B + sqrt(max(0.0f, 1.0f - sqr(t) - sqr(b))) * N};

    // transforming the normal back to the ellipsoid configuration
    return normalize(vec3{ax * nh.x, ay * nh.y, max(0.0f, nh.z)});
}

__both__ vec3 sample_gtr2_bsdf(float a, vec2 const& u)
{
    auto const theta{atan((a * sqrt(u[0])) / sqrt(1.0f - u[0]))};
    auto const phi{two_pi * u[1]};
    return to_sphere_coordinates(theta, phi);
}

/// \brief Specular lobe describe with mircofacets using Trowbridge-Reitz distribution [0][2][3]
/// \details
///     Disney proposes in the 2012 paper a isotropic and anisotropic version of the distribution.
///     In later papers Heitz showed show how to sample the anisotropic version correctly.
///     Besides that, the Trowbridge-Reitz, or GGX model, had a great adoption by the industry.
///     Therefore, it is better to sample the visible normals of the microfacet distribution because
///     it shows faster convergence. As it was discussed in [0], the geometric attenuation should
///     consider height and direction to counter the over-estimation when using the separable version.
/// \note
///     The original version of the specular in 2012 is different but Disney different. This
///     differs in implementation that one uses the visible normals and correlated geometric attenuation.
///     The two reasons are: the modern version are faster and better explained in [0] and [2].
/// \return reflectance
__both__ vec3 disney_specular_brdf_lobe(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    auto wh{wi + wo};
    if (all_zero(wh)) return vec3{0.0f};
    wh = owl::normalize(wh);

    // this part was only briefly mentioned but never really shown in the paper
    auto const lum = luminance(m.base_color);
    auto const c_tint = lum > 0.0f ? m.base_color / lum : vec3{1.0f};
    // 2015 paper Disney mentioned that the F0 is 0.08 * specular
    // this implied an ior for common materials in the range between [1, 1.8]
    // to calculate the used the impl. in brdf explorer
    auto const c_spec = lerp(m.specular * .08f * lerp(vec3{1.0f}, c_tint, m.specular_tint),
            m.base_color, m.metallic);

    auto const a{roughness_to_alpha(m.roughness, m.anisotropic)};
    auto const ax{a.x}, ay{a.y};

    auto const d_term{d_gtr_2(wh, ax, ay)};
    auto const g_term{g2_smith_correlated(wo, wi, ax, ay)};

    auto const f_term{lerp(c_spec, {1.0f}, schlick_weight(dot(wo, wh)))};

    // not using cos_theta(wi) because one multiplies later the love with it
    return m.metallic * d_term * g_term * f_term /
           (4.0f * abs(cos_theta(wo)));
}

inline __both__ float disney_specular_brdf_pdf(material_data const& m, vec3 const& wo,
                                               vec3 const& wh)
{
    auto const a{roughness_to_alpha(m.roughness, m.anisotropic)};
    auto const ax{a.x}, ay{a.y};
    return d_gtr_2(wh, ax, ay) * g1_smith(wo, ax, ay) * max(0.0f, dot(wo, wh)) /
           (4.0f * cos_theta(wo));
    // used for ndf sampling
    // return d_gtr_2(wh, ax, ay) * abs(cos_theta(wh));
}

inline __both__ vec3 disney_specular_brdf_sample(material_data const& m, vec3 const& wo, random& random,
                                                 vec3& wi, float& pdf)
{
    // TODO: CHECK ANISOTROPY BECAUSE OF SKEWED ROTATION
    auto const a{roughness_to_alpha(m.roughness, m.anisotropic)};
    auto const ax{a.x}, ay{a.y};
    auto const wh = sample_gtr2_vndf(wo, ax, ay, random.rng<vec2>());
    wi = reflect(wo, wh);
    pdf = disney_specular_brdf_pdf(m, wo, wh);
    return disney_specular_brdf_lobe(m, wo, wi);
}

/// \brief Specular lobe describe the transmission using Trowbridge-Reitz distribution [0][2][3]
/// \details
///     In 2015 Disney explained that the BRDF model is being extended with a BTDF lobe. For this
///     they used Microfacet to model transmissions.
///
__both__ vec3 disney_specular_bsdf_lobe(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    if (same_hemisphere(wo, wi)) return 0;  // transmission only

    auto const cosThetaO = cos_theta(wo);
    auto const cosThetaI = cos_theta(wi);
    if (cosThetaI == 0 || cosThetaO == 0) return vec3{0.0f};

    // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
    auto const eta{cos_theta(wo) > 0.0f ? m.ior : 1.0f / m.ior};
    auto wh = normalize(wo + wi * eta);
    if (wh.z < 0) wh = -wh;

    // Same side?
    if (dot(wo, wh) * dot(wi, wh) > 0) return vec3{0.0f};

    auto const F = 1.0f;

    auto const sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
    auto const factor = 1.0f / eta;

    auto const a{roughness_to_alpha(m.roughness, m.anisotropic)};

    return (vec3 (1.f) - F) *
           abs(d_gtr_2(wh, a.x, a.y) * g2_smith_correlated(wo, wi, wh, a.x, a.y) * eta * eta *
                   abs(dot(wi, wh)) * abs(dot(wo, wh)) * factor * factor /
                    (cosThetaI * cosThetaO * sqrtDenom * sqrtDenom));
}

inline __both__ float disney_specular_bsdf_pdf(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    if (same_hemisphere(wo, wi)) return 0;
    // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
    auto const eta{cos_theta(wo) > 0.0f ? 1.0f / m.ior : m.ior};
    auto const wh = normalize(wo + wi * eta);

    if (dot(wo, wh) * dot(wi, wh) > 0) return 0;

    auto const a{roughness_to_alpha(m.roughness, m.anisotropic)};
    // Compute change of variables _dwh\_dwi_ for microfacet transmission
    float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
    float dwh_dwi =
            std::abs((eta * eta * dot(wi, wh)) / (sqrtDenom * sqrtDenom));
    return dwh_dwi * d_gtr_2(wh, a.x, a.y);
}

inline __both__ vec3 disney_specular_bsdf_sample(material_data const& m, vec3 const& wo, random& random,
                                                 vec3& wi, float& pdf)
{
    if (wo.z == 0.0f)
    {
        wi = vec3{0.0f};
        pdf = 0.0f;
        return vec3{0.0f};
    }

    // Sampling the specular bsdf lobe requires a different strategy than the brdf specular lobe.
    // Using the VNDF is strangely insufficient for the transmission lobe. One has to use the
    // eq. (35) and (36) from [4] in order to get correct results.

    // auto const a{roughness_to_alpha(m.roughness, m.anisotropic)};
    // auto wh = sample_gtr2_vndf(wo, a.x, a.y, random.rng<vec2>());
    // printf("%f %f %f\n", test.x, test.y, test.z);

    // auto wh = vec3{0,0,1};
    auto wh = sample_gtr2_bsdf(roughness_to_alpha(m.roughness), random.rng<vec2>());
    if (!same_hemisphere(wo, wh)) wh = -wh;

    auto const eta_i{cos_theta(wo) > 0.0f ? 1.0f : m.ior};
    auto const eta_t{cos_theta(wo) > 0.0f ? m.ior : 1.0f};
    auto const eta = eta_i / eta_t;
    auto const f = fresnel_equation(wo, wh, eta_i, eta_t);

    if (!refract(wo, wh, eta, wi) || f > random())
        return disney_specular_brdf_sample(m, wo, random, wi, pdf);
    wi = normalize(wi);

    pdf = 1.0f;
    return 1.0f;
}

#endif //PATH_TRACER_DISNEY_SPECULAR_CUH
