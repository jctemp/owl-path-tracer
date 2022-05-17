#ifndef PATH_TRACER_HPP
#define PATH_TRACER_HPP
#pragma once

#include "device.hpp"
#include "materials/Lambert.hpp"
#include "materials/Disney.hpp"

#include <owl/owl_device.h>
#include <optix_device.h>


using namespace owl;

extern launch_params_data optixLaunchParams;

PT_DEVICE Float2 uvOnSphere(Float3 n)
{
	auto u = 0.5f + atan2(n.x, n.z) / (2 * M_PI);
	auto v = 0.5f + asin(n.y) / M_PI;
	return Float2{ u,v };
}


PT_DEVICE Float3 sampleEnvironment(Float3 dir)
{
	vec2f tc{ uvOnSphere(dir) };
	owl::vec4f const texColor{
		tex2D<float4>(optixLaunchParams.environment_map, tc.x, tc.y) };
	return vec3f{ texColor };
}

PT_DEVICE Float3 tracePath(owl::Ray& ray, Random& random)
{
	auto& LP{ optixLaunchParams };

	// hold total sum of accumulated radiance
	Float3 L{ 0.0f };

	// hold the path throughput weight (f * cos(theta)) / pdf
	// => current implementation has f and cos already combined
	Float3 beta{ 1.0f };

	InterfaceStruct is;
	material_data ms;

	PerRayData prd{ random, ScatterEvent::NONE, &is, &ms };


	for (Int depth{ 0 }; depth < LP.max_path_depth; ++depth)
	{
		/* FIND INTERSECTION */
		owl::traceRay(LP.world, ray, prd);


		/* TERMINATE PATH AND SAMPLE ENVIRONMENT*/
		if (prd.scatterEvent == ScatterEvent::MISS)
		{
			Float3 li{ 0.0f };
			if (!LP.use_environment_map)
				li = 0.0f;
			else if (optixLaunchParams.environment_map)
				li = sampleEnvironment(ray.direction);
			else
				li = mix(Float3{ 1.0f }, Float3{ 0.5f, 0.7f, 1.0f }, { 0.5f *
					(ray.direction.y + 1.0f) });

			L += li * beta;
			break;
		}


		/* PREPARE MESH FOR CALCULATIONS */
		Float3& P{ is.P }, N{ is.N }, V{ is.V }, Ng{ is.Ng };
		Float3 T{}, B{};
		onb(N, T, B);


		/* TERMINATE PATH AND SAMPLE LIGHT */
		if (is.lightId >= 0)
		{
			// TODO: SAMPLE MESH LIGHTx
			light_data light{};
			GET(light, light_data, LP.light_buffer, is.lightId);
			Float3 emission{ light.color * light.intensity };
			L += emission * beta;
			break;
		}


		/* SAMPLE BRDF OR PHASE FUNCTION */
		Float3 L{ 0.0f };
		toLocal(T, B, N, V);

		{
			material_data material{};
			if (is.matId >= 0) GET(material, material_data, LP.material_buffer, is.matId);


			Float pdf{ 0.0f };
			Float3 bsdf{ 0.0f };

			sampleDisneyBSDF(material, V, L, prd.random, bsdf, pdf);

			// end path if impossible
			if (pdf <= 0.0f)
				break;

			beta *= bsdf / pdf;
		}

		toWorld(T, B, N, L);


		/* SAMPLE DIRECT LIGHTS */
		if (LP.light_buffer.count != 0)
		{
			// MAY BE INTRODUCE DOME SAMPLING LATER
			Int randMax{ LP.light_buffer.count };
			Int randId{ (Int)min(prd.random() * randMax, randMax - 1.0f) };


			light_data lights{};



		}

		/* TERMINATE PATH IF RUSSIAN ROULETTE  */
		Float betaMax{ max(beta.x, max(beta.y, beta.z)) };
		if (depth > 3) {
			Float q{ max(.05f, 1 - betaMax) };
			if (prd.random() < q) break;
			beta /= 1 - q;

			ASSERT(isinf(beta.y), "Russian Roulette caused beta to have inf. component");
		}


		ray = owl::Ray{ P,L,T_MIN, T_MAX };
	}

	return L;
}

#endif // !PATH_TRACER_HPP
