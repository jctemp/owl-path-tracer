
#ifndef PATH_TRACER_PATH_TRACER_HPP
#define PATH_TRACER_PATH_TRACER_HPP

#include "device.hpp"
// #include "materials/Lambert.hpp"
#include "materials.hpp"
//#include "disney.hpp"

#include "owl/include/owl/owl_device.h"

#include "macros.hpp"

using namespace owl;

extern __constant__ launch_params_data optixLaunchParams;

__device__ vec2 uvOnSphere(vec3 n)
{
    float const u{0.5f + atan2(n.x, n.z) / (2.0f * pi)};
    float const v{0.5f + asin(n.y) / pi};
    return vec2{u, v};
}


__device__ vec3 sample_environment(vec3 dir)
{
    vec2 tc{uvOnSphere(dir)};
    owl::vec4f const texColor{
            tex2D<float4>(optixLaunchParams.environment_map, tc.x, tc.y)};
    return vec3f{texColor};
}

using radiance_ray = owl::RayT<0, 2>;
using shadow_ray = owl::RayT<1, 2>;

__device__ float power_heuristic(float n_f, float pdf_f, float n_g, float pdf_g)
{
    float f = n_f * pdf_f;
    float g = n_g * pdf_g;
    return (f * f) / (f * f + g * g);
}

__device__ vec3 tracePath(radiance_ray& ray, Random& random)
{
    auto& LP{optixLaunchParams};

    // hold total sum of accumulated radiance
    vec3 L{0.0f};

    // hold the path throughput weight (f * cos(theta)) / pdf
    // => current implementation has f and cos already combined
    vec3 beta{1.0f};

    interface_data is;
    material_data ms;

    per_ray_data prd{random, scatter_event::NONE, &is, &ms};

    for (int32_t depth{0}; depth < LP.max_path_depth; ++depth)
    {
        /* FIND INTERSECTION */
        owl::traceRay(LP.world, ray, prd);


        /* TERMINATE PATH AND SAMPLE ENVIRONMENT*/
        if (prd.scatterEvent == scatter_event::MISS)
        {
            vec3 li{0.0f};
            if (!LP.use_environment_map)
                li = 0.0f;
            else if (LP.environment_map)
                li = sample_environment(ray.direction);
            else
                li = lerp(vec3{1.0f}, vec3{0.5f, 0.7f, 1.0f}, 0.5f * (ray.direction.y + 1.0f));

            L += li * beta;
            break;
        }


        /* PREPARE MESH FOR CALCULATIONS */
        vec3& p{is.P}, n{is.N}, wo{is.V};
        vec3 T{}, B{};
        onb(n, T, B);


        /* TERMINATE PATH AND SAMPLE LIGHT */
        if (is.lightId >= 0)
        {
            // TODO: SAMPLE LIGHT
            light_data light{};
            get_data(light, LP.light_buffer, is.lightId, light_data);
            vec3 emission{light.color * light.intensity};
            L += emission * beta;
            break;
        }


        /* SAMPLE BRDF OR PHASE FUNCTION */
        vec3 wi{0.0f};
        to_local(T, B, n, wo);

        material_data material{};
        if (is.matId >= 0) { get_data(material, LP.material_buffer, is.matId, material_data); }

        auto pdf{0.0f};
        auto f{vec3{0.0f}};

        switch (material.type)
        {
            case material_data::type::disney:
                sample_disney_bsdf(material, wo, prd.random, wi, f, pdf);
                break;
            case material_data::type::lambert:
                sample_lambert(material, wo, prd.random, wi, f, pdf);
                break;
            case material_data::type::disney_diffuse:
                sample_disney_diffuse(material, wo, prd.random, wi, f, pdf);
                break;
            case material_data::type::disney_subsurface:
                sample_disney_subsurface(material, wo, prd.random, wi, f, pdf);
                break;
            case material_data::type::disney_retro:
                sample_disney_retro(material, wo, prd.random, wi, f, pdf);
                break;
            case material_data::type::disney_sheen:
                sample_disney_sheen(material, wo, prd.random, wi, f, pdf);
                break;
            case material_data::type::disney_clearcoat:
                sample_disney_clearcoat(material, wo, prd.random, wi, f, pdf);
                break;
            case material_data::type::disney_microfacet:
                sample_disney_microfacet(material, wo, prd.random, wi, f, pdf);
                break;
            default:
                printf("unknown material type\n");
                break;
        }

        // end path if impossible
        if (pdf <= 0.0f)
            break;

        beta *= f / pdf;

        to_world(T, B, n, wi);

        /* SAMPLE DIRECT LIGHTS */
        if (false)
        {
            // TODO: GET LIGHT DATA FROM GLOBAL BUFFER
            auto const light_position{2.0f};
            auto const light_intensity{1.0f};

            // CALCULATE LIGHT SPATIAL METRICS
            auto const light_wi{normalize(light_position - p)};
            auto const light_pdf{1.0f};
            auto const light_power{light_intensity * 4 * pi};
            auto const light_li{light_intensity / sqr(light_position - p)};

            // VISIBILITY TEST
            shadow_ray light_ray{p, light_wi, t_min, light_position + t_min};

            bool visible{false};
            owl::traceRay(LP.world, light_ray, visible,
                    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
                    | OPTIX_RAY_FLAG_DISABLE_ANYHIT
                    | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);

            auto surface_f{ vec3{0.0f} };
            auto surface_pdf{ 0.0f };

            if (visible && light_pdf > 0.0)
            {
                // TODO: add additional branch to evaluate subsurface scattering

                // TODO: replace f_lambert with f_disney
                surface_f =  f_lambert(material, wo, light_wi) * owl::abs(owl::dot(light_wi, n));
                surface_pdf = pdf_lambert(material, wo, light_wi);

                if (!all_zero(surface_f) && !all_zero(light_li))
                {
                    auto const weight = power_heuristic(1, light_pdf, 1, surface_pdf);
                    L += surface_f * light_li * weight / surface_pdf;
                }
            }
        }

        /* TERMINATE PATH IF RUSSIAN ROULETTE  */
        auto const beta_max{owl::max(beta.x, owl::max(beta.y, beta.z))};
        if (depth > 3)
        {
            float q{owl::max(.05f, 1 - beta_max)};
            if (prd.random() > q) break;
        }

        ray = radiance_ray{p, wi, t_min, t_max};
    }

    return L;
}

#endif //PATH_TRACER_PATH_TRACER_HPP
