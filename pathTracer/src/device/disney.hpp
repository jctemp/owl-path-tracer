
#ifndef PATH_TRACER_DISNEY_HPP
#define PATH_TRACER_DISNEY_HPP

#include "math.hpp"
#include "types.hpp"
#include "device.hpp"
#include "sample_methods.hpp"

// * https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// * https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf

/**
 * @brief TODO: add description
 * @param mat material data for the current object
 * @param wo outgoing direction (normalized), corresponds to L in the paper
 * @param wi incoming direction (normalized), corresponds to V in the paper
 * @return vec3 the reflectance/ throughput of the path
 */
__both__ vec3 f_disney(material_data const& mat, vec3 const& wo, vec3 const& wi)
{
    if (!same_hemisphere(wo, wi)) return vec3{0.0f };
    return mat.baseColor * inv_pi;
}

/**
 * @brief TODO: add description
 * @param mat material data for the current object
 * @param wo outgoing direction (normalized), corresponds to L in the paper
 * @param wi incoming direction (normalized), corresponds to V in the paper
 * @return float the pdf of the path for the sampled direction
 */
__both__ float pdf_disney(material_data const& mat, vec3 const& wo, vec3 const& wi)
{
    if (!same_hemisphere(wo, wi)) return 0.0f;
    return pdf_cosine_hemisphere(wo, wi);
}

/**
 * @brief TODO: add description
 * @param mat material data for the current object
 * @param wo outgoing direction (normalized), corresponds to L in the paper
 * @param wi incoming direction (normalized), corresponds to V in the paper
 * @param rng random number generator for sampling
 * @param wi the sampled direction (out parameter)
 * @param f the reflectance/ throughput of the path (out parameter)
 * @param pdf of the path for the sampled direction (out parameter)
 */
__both__ void sample_disney(material_data const& mat, vec3 const& wo, Random& rng, vec3& wi, vec3& f, float& pdf)
{
    wi = sample_cosine_hemisphere(rng.random<vec2>());
    pdf = pdf_disney(mat, wo, wi);
    f = f_disney(mat, wo ,wi);
}

#endif //PATH_TRACER_DISNEY_HPP
