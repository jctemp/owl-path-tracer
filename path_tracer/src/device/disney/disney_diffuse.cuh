
#ifndef PATH_TRACER_DISNEY_DIFFUSE_CUH
#define PATH_TRACER_DISNEY_DIFFUSE_CUH

#include "sample_methods.hpp"
#include "types.hpp"
#include "macros.hpp"
#include "disney_helper.cuh"

/// \brief Disney diffuse lobe. A subsurface approximation for small
///     scattering distances.
/// \details
///     This function describes the diffuse lobe of the Disney BSDF model.
///     It is a refactored version of the presented implementation in 2012
///     to allow for the substitution of the lambertian part with subsurface
///     scattering. The model shows same results as the diffusion model for
///     small scattering distances.
/// \note
///     As stated in the Disney Paper in 2015, the diffuse lobe is not
///     physically correct in the sense that it is not energy conserving.
///     This leads to unnatural appearance for non-plausible values greater
///     equal 1 for the base_color (albedo parameter).
/// \param wo The outgoing direction.
/// \param wi The sampled incident direction.
/// \returns The reflectance of this lobe. Color is already accounted for.
__both__ vec3 eval_disney_diffuse(material_data const& m, vec3 const& wo, vec3 const& wh, vec3 const& wi, float& pdf)
{
    /// 2015 - refactored diffuse component                                         <br>
    ///                                                                             <br>
    ///     fd = (base_color / pi) * (1 - 0.5 * Fl) * (1 - 0.5 * Fv) + fr           <br>
    ///     fr = (base_color / pi) * rr (Fl + Fv + Fl * Fv * (rr - 1.0f))           <br>
    ///     rr = 2 * roughness * cos^2(theta_d) where theta_d = wh * wi = wh * wo   <br>
    ///                                                                             <br>

    auto const cos_theta_o{cos_theta(wo)};
    auto const cos_theta_i{cos_theta(wi)};
    auto const fresnel_o{schlick_weight(cos_theta_o)};
    auto const fresnel_i{schlick_weight(cos_theta_i)};
    auto const lambert{m.base_color * inv_pi};

    auto fd{(1.0f - 0.5f * fresnel_o) * (1.0f - 0.5f * fresnel_i)};

    /// rr is calculated without to obtain the half-vector twice the half vector is <br>
    /// the angle between the two incident directions                               <br>
    ///                                                                             <br>
    ///     2 * cos^2(theta_d) = 1 + cos(2 * theta_d) = 1 + wo * wi                 <br>
    ///                                                                             <br>

    auto const rr{m.roughness * (dot(wo, wi) + 1.0f)};
    auto const fr{rr * (fresnel_i + fresnel_o + fresnel_o * fresnel_i * (rr - 1.0f))};

    pdf = pdf_cosine_hemisphere(wo, wi);

    return lambert * (fd + fr);
}

inline __both__ vec3 sample_disney_diffuse(material_data const& m, vec3 const& wo, random& random,
                                           vec3& wi, float& pdf)
{
    wi = sample_cosine_hemisphere(random.rng<vec2>());
    return eval_disney_diffuse(m, wo, {} ,wi, pdf);
}

#endif //PATH_TRACER_DISNEY_DIFFUSE_CUH
