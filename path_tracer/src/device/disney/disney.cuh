#ifndef PATH_TRACER_DISNEY_CUH
#define PATH_TRACER_DISNEY_CUH

#include "device/disney/disney_diffuse.cuh"
#include "device/disney/disney_specular.cuh"
#include "device/disney/disney_sheen.cuh"
#include "device/disney/disney_clearcoat.cuh"

#define DISNEY_SAMPLED_LOBE_NONE -1
#define DISNEY_SAMPLED_LOBE_DIFFUSE 0
#define DISNEY_SAMPLED_LOBE_CLEARCOAT 1
#define DISNEY_SAMPLED_LOBE_METALLIC 2
#define DISNEY_SAMPLED_LOBE_GLASS 3

__both__ void calculate_pdf_of_lobes(material_data const& m, float& p_metallic, float& p_diffuse,
                                     float& p_clearcoat, float& p_glass)
{
    auto diffuse_weight = (1.0f - m.specular_transmission) * (1.0f - m.metallic);
    auto metallic_weight = m.metallic;
    auto clearcoat_weight = .25f * m.clearcoat;
    auto glass_weight = (1.0f - m.metallic) * m.specular_transmission;

    float factor = 1.0f / (metallic_weight + glass_weight + diffuse_weight + clearcoat_weight);

    p_metallic = metallic_weight * factor;
    p_glass = glass_weight * factor;
    p_diffuse = diffuse_weight * factor;
    p_clearcoat = clearcoat_weight * factor;
}

__both__ vec3 sample_disney(material_data const& m, vec3 const& wo, random& random,
                            vec3& wi, float& pdf, int& sampled_lobe)
{
    float p_metallic;
    float p_diffuse;
    float p_clearcoat;
    float p_glass;
    calculate_pdf_of_lobes(m, p_metallic, p_diffuse, p_clearcoat, p_glass);

    bool force_btdf{cos_theta(wo) < 0.0f && sampled_lobe == DISNEY_SAMPLED_LOBE_GLASS};

    float p = random();
    vec3 f{};
    if(!force_btdf && p <= p_metallic)
    {
        f = sample_disney_specular_brdf(m, wo, random, wi, pdf);
        sampled_lobe = DISNEY_SAMPLED_LOBE_METALLIC;
    }
    else if(!force_btdf && p > p_metallic && p <= (p_metallic + p_clearcoat))
    {
        f = sample_disney_clearcoat(m, wo, random, wi, pdf);
        sampled_lobe = DISNEY_SAMPLED_LOBE_CLEARCOAT;
    }
    else if(!force_btdf && p > p_metallic + p_clearcoat && p <= (p_metallic + p_clearcoat + p_diffuse))
    {
        f = sample_disney_diffuse(m, wo, random, wi, pdf);
        sampled_lobe = DISNEY_SAMPLED_LOBE_DIFFUSE;
    }
    else if(force_btdf || p_glass >= 0.0f)
    {
        f = sample_disney_specular_bsdf(m, wo, random, wi, pdf);
        sampled_lobe = DISNEY_SAMPLED_LOBE_GLASS;
    }

    return f + eval_disney_sheen(m, wo, wi);
}


#endif //PATH_TRACER_DISNEY_CUH
