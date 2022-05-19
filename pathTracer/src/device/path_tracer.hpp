
#ifndef PATH_TRACER_PATH_TRACER_HPP
#define PATH_TRACER_PATH_TRACER_HPP

#include "device.hpp"
#include "materials/Lambert.hpp"
#include "materials/Disney.hpp"

#include "owl/include/owl/owl_device.h"

#include "macros.hpp"

using namespace owl;

extern __constant__ launch_params_data optixLaunchParams;

__device__ vec2 uvOnSphere(vec3 n)
{
    float const u{ 0.5f + atan2(n.x, n.z) / (2.0f * pi) };
    float const v{ 0.5f + asin(n.y) / pi };
    return vec2{ u,v };
}


__device__ vec3 sampleEnvironment(vec3 dir)
{
    vec2 tc{ uvOnSphere(dir) };
    owl::vec4f const texColor{
            tex2D<float4>(optixLaunchParams.environment_map, tc.x, tc.y) };
    return vec3f{ texColor };
}

__device__ vec3 tracePath(owl::Ray& ray, Random& random)
{
    auto& LP{ optixLaunchParams };

    // hold total sum of accumulated radiance
    vec3 L{ 0.0f };

    // hold the path throughput weight (f * cos(theta)) / pdf
    // => current implementation has f and cos already combined
    vec3 beta{ 1.0f };

    InterfaceStruct is;
    material_data ms;

    per_ray_data prd{ random, ScatterEvent::NONE, &is, &ms };


    for (int32_t depth{ 0 }; depth < LP.max_path_depth; ++depth)
    {
        /* FIND INTERSECTION */
        owl::traceRay(LP.world, ray, prd);


        /* TERMINATE PATH AND SAMPLE ENVIRONMENT*/
        if (prd.scatterEvent == ScatterEvent::MISS)
        {
            vec3 li{ 0.0f };
            if (!LP.use_environment_map)
                li = 0.0f;
            else if (optixLaunchParams.environment_map)
                li = sampleEnvironment(ray.direction);
            else
                li = lerp(vec3{ 1.0f }, vec3{ 0.5f, 0.7f, 1.0f },  0.5f * (ray.direction.y + 1.0f) );

            L += li * beta;
            break;
        }


        /* PREPARE MESH FOR CALCULATIONS */
        vec3& P{ is.P }, N{ is.N }, V{ is.V };
        vec3 T{}, B{};
        onb(N, T, B);


        /* TERMINATE PATH AND SAMPLE LIGHT */
        if (is.lightId >= 0)
        {
            // TODO: SAMPLE MESH LIGHTx
            light_data light{};
            get_data(light, LP.light_buffer, is.lightId, light_data);
            vec3 emission{ light.color * light.intensity };
            L += emission * beta;
            break;
        }


        /* SAMPLE BRDF OR PHASE FUNCTION */
        vec3 L{ 0.0f };
        to_local(T, B, N, V);

        {
            material_data material{};
            if (is.matId >= 0) get_data(material, LP.material_buffer, is.matId, material_data);


            float pdf{ 0.0f };
            vec3 bsdf{ 0.0f };

            sampleDisneyBSDF(material, V, L, prd.random, bsdf, pdf);

            // end path if impossible
            if (pdf <= 0.0f)
                break;

            beta *= bsdf / pdf;
        }

        to_world(T, B, N, L);

        /* SAMPLE DIRECT LIGHTS */
        if (LP.light_buffer.count != 0)
        {
            // MAY BE INTRODUCE DOME SAMPLING LATER
            int32_t rand_max{ static_cast<int32_t>(LP.light_buffer.count)};
            int32_t rand_id{ static_cast<int32_t>(min(prd.random() * rand_max, rand_max - 1.0f))};
            light_data lights{};
        }

        /* TERMINATE PATH IF RUSSIAN ROULETTE  */
        float betaMax{ max(beta.x, max(beta.y, beta.z)) };
        if (depth > 3) {
            float q{ max(.05f, 1 - betaMax) };
            if (prd.random() < q) break;
            beta /= 1 - q;
            assert_condition(isinf(beta.y), "Russian Roulette caused beta to have inf. component");
        }

        ray = owl::Ray{ P,L,t_min, t_max };
    }

    return L;
}

#endif //PATH_TRACER_PATH_TRACER_HPP
