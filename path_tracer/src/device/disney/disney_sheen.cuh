#ifndef PATH_TRACER_DISNEY_SHEEN_CUH
#define PATH_TRACER_DISNEY_SHEEN_CUH

#include "sample_methods.hpp"
#include "types.hpp"
#include "macros.hpp"
#include "disney_helper.cuh"

/// \brief Disney sheen component. An optional part of the BSDF model to model light at
///     grazing angles which behave Fresnel-like.
/// \details
///     This component is based on observations of fabric samples and should add a missing
///     effect of the diffuse and specular reflectance lobe at grazing angles.
/// \returns Extra reflectance for grazing angles.
__both__ vec3 disney_sheen_component(material_data const& m, vec3 const& wo, vec3 const& wi)
{
    if (m.sheen <= 0.0f) return vec3{0.0f};

    /// Handle the possible degenerated half-vector resulting in a zero dot product
    auto wh{wi + wo};
    if (all_zero(wh)) return vec3{0.0f};
    wh = owl::normalize(wh);

    ///                                                                             <br>
    /// Sheen tint has not really been touched in the Disney papers. As it is not   <br>
    /// that relevant as an optional component. Revising the BRDF Explorer one can  <br>
    /// find out how to calculate the sheen tint. Difference to this implementation <br>
    /// is that one uses the base_color directly instead of a linear version        <br>
    ///                                                                             <br>

    auto const lum{luminance(m.base_color)};
    auto const cos_theta_d{owl::dot(wi, wh)};
    auto const tint{(lum > 0.0f) ? m.base_color / lum : vec3{1.0f}};

    return lerp(vec3{1.0f}, tint, m.sheen_tint) * schlick_weight(cos_theta_d);
}


#endif //PATH_TRACER_DISNEY_SHEEN_CUH
