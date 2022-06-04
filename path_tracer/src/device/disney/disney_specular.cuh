#ifndef PATH_TRACER_DISNEY_SPECULAR_CUH
#define PATH_TRACER_DISNEY_SPECULAR_CUH

#include "sample_methods.hpp"
#include "types.hpp"
#include "macros.hpp"
#include "disney_helper.cuh"

// References:
// - Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs [0]
// - Average irregularity representation of a rough surface for ray reflection [1]


/// Because of using the revised version of the smith g term one adopts the
/// eq. (85) from [0]
/// wm here is the normal vector of the microfacet [1]
float d_gtr_2(vec3 const& wm, float ax, float ay)
{
    auto const tan2_theta {sqr(tan_theta(wm))};
    if (isinf(tan2_theta)) return 0.0f;
    auto const cos4_theta{sqr(sqr(cos_theta(wm)))};
    auto const e{1.0f + tan2_theta * (sqr(cos_phi(wm)) / sqr(ax) + sqr(sin_phi(wm)) / sqr(ay))};
    return 1.0f / (pi * ax * ay * cos4_theta * sqr(e));
}


/// Addenda Specular G revisited => uses now derived version by Heitz
/// eq. (43) from [0]
float g_smith_gtr(vec3 const& w, float ax, float ay)
{
    auto const abs_tan_theta {owl::abs(tan_theta(w))};
    if (isinf(abs_tan_theta)) return 0.0f;
    // eq. (80) from [0] // TODO: CHECK IF COS_PHI IS CORRECT
    auto const alpha0{sqrt(sqr(w.x * ax) + sqr(w.y * ay))};
    // auto const alpha0{sqrt(sqr(cos_phi(w) * ax) + sqr(sin_phi(w) * ay))};
    auto const a{1.0f / (alpha0 * abs_tan_theta)};
    // eq. (86) from [0]
    auto const lambda{(-1.0f + sqrt(1.0f + 1.0f / sqr(a))) / 2.0f};
    return 1.0f / (1.0f + lambda);
}


__both__ vec3 disney_specular_brdf_lobe(material_data const& m, vec3 const& wo, vec3 const& wi, vec3 const& wm)
{
    //TODO: 
}


#endif //PATH_TRACER_DISNEY_SPECULAR_CUH
