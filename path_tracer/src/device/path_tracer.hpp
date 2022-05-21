
#ifndef PATH_TRACER_PATH_TRACER_HPP
#define PATH_TRACER_PATH_TRACER_HPP

#include <owl/owl_device.h>

#include "device.hpp"
#include "materials.hpp"
#include "lights.hpp"
#include "macros.hpp"

using radiance_ray = owl::RayT<0, 2>;
using shadow_ray = owl::RayT<1, 2>;

extern __constant__ launch_params_data optixLaunchParams;

__device__ vec2 uvOnSphere(vec3 n)
{
    auto const u{0.5f + atan2(n.x, n.z) / (2.0f * pi)};
    auto const v{0.5f + asin(n.y) / pi};
    return vec2{u, v};
}

__device__ vec3 sample_environment(vec3 dir)
{
    vec2 tc{uvOnSphere(dir)};
    owl::vec4f const texColor{
            tex2D<float4>(optixLaunchParams.environment_map, tc.x, tc.y)};
    return vec3{texColor};
}

__device__ vec3 trace_path(radiance_ray& ray, Random& random)
{
    auto& launch_params{optixLaunchParams};

    // hold total sum of accumulated radiance
    vec3 radiance{0.0f};

    // hold the path throughput weight (f * cos(theta)) / pdf
    // => current implementation has f and cos already combined
    vec3 beta{1.0f};

    interface_data is;
    material_data ms;

    per_ray_data prd{random, scatter_event::NONE, &is, &ms};

    for (int32_t depth{0}; depth < launch_params.max_path_depth; ++depth)
    {
        /* FIND INTERSECTION */
        owl::traceRay(launch_params.world, ray, prd);


        /* TERMINATE PATH AND SAMPLE ENVIRONMENT*/
        if (prd.scatterEvent == scatter_event::MISS)
        {
            vec3 li{0.0f};
            if (!launch_params.use_environment_map)
                li = 0.0f;
            else if (launch_params.environment_map)
                li = sample_environment(ray.direction);
            else
                li = lerp(vec3{1.0f}, vec3{0.5f, 0.7f, 1.0f}, 0.5f * (ray.direction.y + 1.0f));

            radiance += li * beta;
            break;
        }


        /* PREPARE MESH FOR CALCULATIONS */
        auto const& hit_p{is.P}, hit_n{is.N}, world_wo{is.V};

        vec3 T{}, B{};
        onb(hit_n, T, B);


        /* TERMINATE PATH AND SAMPLE LIGHT */
        if (is.lightId >= 0)
        {
            // TODO: SAMPLE LIGHT
            light_data light{};
            get_data(light, launch_params.light_buffer, is.lightId, light_data);
            vec3 emission{light.color * light.intensity};
            radiance += emission * beta;
            break;
        }


        /* SAMPLE BRDF */
        material_data material{};
        if (is.matId >= 0) { get_data(material, launch_params.material_buffer, is.matId, material_data); }

        auto const local_wo{to_local(T, B, hit_n, world_wo)};
        auto local_wi{vec3{0.0f}};
        auto hit_pdf{0.0f};
        auto hit_f{vec3{0.0f}};

        switch (material.type)
        {
            case material_data::type::disney:
                sample_disney_bsdf(material, local_wo, prd.random, local_wi, hit_f, hit_pdf);
                break;
            case material_data::type::lambert:
                sample_lambert(material, local_wo, prd.random, local_wi, hit_f, hit_pdf);
                break;
            case material_data::type::disney_diffuse:
                sample_disney_diffuse(material, local_wo, prd.random, local_wi, hit_f, hit_pdf);
                break;
            case material_data::type::disney_subsurface:
                sample_disney_subsurface(material, local_wo, prd.random, local_wi, hit_f, hit_pdf);
                break;
            case material_data::type::disney_retro:
                sample_disney_retro(material, local_wo, prd.random, local_wi, hit_f, hit_pdf);
                break;
            case material_data::type::disney_sheen:
                sample_disney_sheen(material, local_wo, prd.random, local_wi, hit_f, hit_pdf);
                break;
            case material_data::type::disney_clearcoat:
                sample_disney_clearcoat(material, local_wo, prd.random, local_wi, hit_f, hit_pdf);
                break;
            case material_data::type::disney_microfacet:
                sample_disney_microfacet(material, local_wo, prd.random, local_wi, hit_f, hit_pdf);
                break;
            default:
                printf("unknown material type\n");
                break;
        }

        /* TERMINATE PATH IF IMPOSSIBLE */
        if (hit_pdf <= 1E-5f)
            break;

        beta *= hit_f / hit_pdf;

        auto const world_wi{to_world(T, B, hit_n, local_wi)};

        /* SAMPLE DIRECT LIGHTS */
        if (false)
        {
            // TODO: GET LIGHT DATA FROM GLOBAL BUFFER
            auto const light_position{2.0f};
            auto const light_intensity{1.0f};

            // CALCULATE LIGHT SPATIAL METRICS
            auto const light_wi{normalize(light_position - hit_p)};
            auto const light_pdf{1.0f};
            auto const light_power{light_intensity * 4 * pi};
            auto const light_li{light_intensity / sqr(light_position - hit_p)};

            // VISIBILITY TEST
            shadow_ray light_ray{hit_p, light_wi, t_min, light_position + t_min};

            bool visible{false};
            owl::traceRay(launch_params.world, light_ray, visible,
                    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
                    | OPTIX_RAY_FLAG_DISABLE_ANYHIT
                    | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);

            auto surface_f{ vec3{0.0f} };
            auto surface_pdf{ 0.0f };

            if (visible && light_pdf > 0.0)
            {
                // TODO: add additional branch to evaluate subsurface scattering

                // TODO: replace f_lambert with f_disney
                surface_f = f_lambert(material, world_wo, light_wi) * owl::abs(owl::dot(light_wi, hit_n));
                surface_pdf = pdf_lambert(material, world_wo, light_wi);

                if (!all_zero(surface_f) && !all_zero(light_li))
                {
                    auto const weight = power_heuristic(1, light_pdf, 1, surface_pdf);
                    radiance += surface_f * light_li * weight / surface_pdf;
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

        ray = radiance_ray{hit_p, world_wi, t_min, t_max};
    }

    return radiance;
}

#endif //PATH_TRACER_PATH_TRACER_HPP
