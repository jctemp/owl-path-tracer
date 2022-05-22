
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

__device__ vec3 trace_path(radiance_ray& ray, random& random, int32_t& samples)
{
    auto& launch_params{optixLaunchParams};

    // hold total sum of accumulated radiance
    vec3 radiance{0.0f};

    // hold the path throughput weight (f * cos(theta)) / pdf
    // => current implementation has f and cos already combined
    vec3 beta{1.0f};

    interface_data is;
    material_data ms;

    per_ray_data prd{random, scatter_event::none, &is, &ms};

    for (int32_t depth{0}; depth < launch_params.max_path_depth; ++depth)
    {
        /* FIND INTERSECTION */
        owl::traceRay(launch_params.world, ray, prd);


        /* TERMINATE PATH AND SAMPLE ENVIRONMENT*/
        if (prd.scatter_event == scatter_event::missed)
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


        /* PREPARE mesh FOR CALCULATIONS */
        auto const& hit_p{is.position}, hit_n{is.normal}, world_wo{is.wo};

        vec3 T{}, B{};
        onb(hit_n, T, B);


        /* SAMPLE BRDF */
        material_data material{};
        if (is.material_id >= 0) { get_data(material, launch_params.material_buffer, is.material_id, material_data); }

        auto hit_pdf{0.0f};
        auto hit_f{vec3{0.0f}};
        auto sampled_type{material_type::none};

        auto const local_wo{to_local(T, B, hit_n, world_wo)};
        auto local_wi{vec3{0.0f}};
        auto local_wh{vec3{0.0f}};
        sample_disney_bsdf(material, local_wo, prd.random,
                local_wi, local_wh, hit_f, hit_pdf, sampled_type);
        auto const world_wi{to_world(T, B, hit_n, local_wi)};


        /* TERMINATE PATH IF IMPOSSIBLE */
        if (hit_pdf <= 1E-5f)
            break;

        beta *= hit_f / hit_pdf;

        /* SAMPLE DIRECT LIGHTS */
        if (sampled_type == material_type::diffuse)
        {
            samples++;

            /* TODO: ALLOW MULTIPLE POINT LIGHTS */
            light_data point_light{};
            get_data(point_light, launch_params.light_buffer, 0, light_data);

            // CALCULATE LIGHT SPATIAL METRICS
            auto const light_wi{normalize(point_light.position - hit_p)};
            auto const light_pdf{1.0f};

            auto const light_distance{owl::length(point_light.position - hit_p)};
            auto const light_li{point_light.color * point_light.intensity / sqr(light_distance)};

            // VISIBILITY TEST
            shadow_ray light_ray{hit_p, light_wi, t_min, light_distance + t_min};

            bool visible{false};
            owl::traceRay(launch_params.world, light_ray, visible,
                    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
                    | OPTIX_RAY_FLAG_DISABLE_ANYHIT
                    | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);

            auto surface_f{vec3{0.0f}};
            auto surface_pdf{0.0f};

            if (visible && light_pdf > 0.0)
            {
                // TODO: add additional branch to evaluate subsurface scattering

                surface_f = f_disney_bsdf(material, local_wo, light_wi, sampled_type) * owl::abs(owl::dot(light_wi, hit_n));
                surface_pdf = pdf_disney_pdf(material, local_wo, light_wi, sampled_type);

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

    if (has_inf(radiance) || has_nan(radiance))
    {
        printf("error\n");
    }

    return radiance;
}

#endif //PATH_TRACER_PATH_TRACER_HPP
