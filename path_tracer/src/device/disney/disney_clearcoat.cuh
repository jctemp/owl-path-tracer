#ifndef PATH_TRACER_DISNEY_CLEARCOAT_CUH
#define PATH_TRACER_DISNEY_CLEARCOAT_CUH

#include "sample_methods.hpp"
#include "types.hpp"
#include "macros.hpp"
#include "disney_helper.cuh"

// References:
// - Physically Based Shading at Disney [0]

/// 2012 Paper [0] - B.2 eq. (4)
inline __both__ float d_gtr1(vec3 const& wh, float alpha)
{
    auto const a2{sqr(alpha)};
    return (a2 - 1.0f) / (pi * logf(a2) * (1.0f + (a2 - 1.0f) * sqr(cos_theta(wh))));
}

/// 2012 Paper [0] - B.2 eq. (2), (5)
inline __both__ vec3 sample_gtr1_ndf(vec3 const& wo, vec3 const& wi, float a, vec2 const& u)
{
    auto const alpha2{sqr(a)};
    auto const cos_theta{sqrt(max(0.0f, (1.0f - powf(alpha2, 1.0f - u[0])) / (1.0f - alpha2)))};
    auto const sin_theta{sqrt(max(0.0f, 1.0f - sqr(cos_theta)))};
    auto const phi{two_pi * u[1]};

    auto wh{to_sphere_coordinates(sin_theta, cos_theta, phi)};
    if (!same_hemisphere(wo, wh)) wh = -wh;
    return wh;
}

/// \brief Secondary BRDF specular lobe to model a clearcoat layer on top of normal BRDF.
/// \details
///     Follows the same ideas of microfacet models. See in the Appendix B of [0] to
///     get an understanding. This lobe has not physical meaning and was introduce to
///     mimic measured data. It does introduces little amount of energy to the system but
///     it is not really handled by Disney due to the "insignificant" amount in respect
///     to the visual contribution. The clearcoat get fixed parameters for IOR 1.5 and
///     roughness 0.25 as it is assumed to be isotropic and non-metallic. The lobes uses
///     for its normal distribution the generalised trowbridge-reitz distribution with a
///     gamma of 1.
__both__ vec3 disney_clearcoat_lobe(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    auto wh{wi + wo};
    if (all_zero(wh)) return vec3{0.0f};
    wh = owl::normalize(wh);

    auto const a{lerp(0.1f, .001f, m.clearcoat_gloss)};
    auto const d_term{d_gtr1(wh, a)};
    auto const f_term{fresnel_schlick(cos_theta(wi), 1.5f)};
    auto const g_term{g2_smith_separable(wo, wi, .25f, .25f)};

    return d_term * g_term * f_term /
           (4.0f * abs(cos_theta(wo)) * abs(cos_theta(wi)));
}

inline __both__ float disney_clearcoat_pdf(vec3 const& wo, vec3 const& wi, float a)
{
    auto wh{wi + wo};
    if (all_zero(wh)) return 0.0f;
    wh = owl::normalize(wh);

    return d_gtr1(wh, a) * cos_theta(wh) / (4.0f * dot(wh, wi));
}

inline __both__ vec3 disney_clearcoat_sample(material_data const& m, vec3 const& wo, random& random,
                                             vec3& wi, float& pdf)
{
    auto const a{lerp(0.1f, .001f, m.clearcoat_gloss)};
    auto const wh{sample_gtr1_ndf(wo, wi, a, random.rng<vec2>())};
    wi = reflect(wo, wh);
    if (!same_hemisphere(wo, wi)) return vec3{0.0f};
    pdf = disney_clearcoat_pdf(wo, wi, a);
    return disney_clearcoat_lobe(m, wo, wi);
}

#endif //PATH_TRACER_DISNEY_CLEARCOAT_CUH
