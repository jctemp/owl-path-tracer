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
    if (alpha >= 1)
        return inv_pi;

    auto const a2{sqr(alpha)};
    return (a2 - 1.0f) / (pi * logf(a2) * (1.0f + (a2 - 1.0f) * sqr(cos_theta(wh))));
}

/// 2012 Paper [0] - B.2 eq. (2), (5)
inline __both__ vec3 sample_gtr1_ndf(vec3 const& wo, vec3 const& wi, float a, vec2 const& u)
{
    auto const alpha2{sqr(a)};
    auto const cos_theta{sqrt(fmax(0.0f, (1.0f - powf(alpha2, 1.0f - u[0])) / (1.0f - alpha2)))};
    auto const sin_theta{sqrt(fmax(0.0f, 1.0f - sqr(cos_theta)))};
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
__both__ vec3 eval_disney_clearcoat(material_data const& m, vec3 const& wo, vec3 const& wh, vec3 const& wi, float& pdf)
{
    if (m.clearcoat <= 0.0f)
    {
        pdf = 0.0f;
        return vec3{0.0f};
    }

    auto const d{d_gtr1(wh, lerp(0.1f, .001f, m.clearcoat_gloss))};
    auto const f{lerp(1.0f, schlick_weight(cos_theta(wi)), 0.04f)};
    auto const g{g2_smith_separable(wo, wi, .25f, .25f)};

    pdf = d  / (4.0f * dot(wh, wi));
    return d * g * f / (4.0f * abs(cos_theta(wo)) * abs(cos_theta(wi)));
}

inline __both__ vec3 sample_disney_clearcoat(material_data const& m, vec3 const& wo, random& random,
                                             vec3& wi, float& pdf)
{
    auto const a{lerp(0.1f, .001f, m.clearcoat_gloss)};

    auto wh{sample_gtr1_ndf(wo, wi, a, random.rng<vec2>())};
    if (dot(wh, wo) < 0.0f) wh = -wh;
    wh = normalize(wh);

    wi = reflect(wo, wh);
    if (!same_hemisphere(wo, wi))
    {
        pdf = 0.0f;
        return vec3{0.0f};
    }

    return eval_disney_clearcoat(m, wo, wh, wi, pdf);
}

#endif //PATH_TRACER_DISNEY_CLEARCOAT_CUH
