#include "Globals.hpp"
#include "materials/Lambert.hpp"
#include "materials/Disney.hpp"
#include "lights/Light.hpp"

#include <owl/owl_device.h>
#include <optix_device.h>

using namespace owl;

__constant__ LaunchParams optixLaunchParams;

DEVICE Float2 uvOnSphere(Float3 n)
{
	auto u = 0.5f + atan2(n.x, n.z) / (2 * M_PI);
	auto v = 0.5f + asin(n.y) / M_PI;
	return Float2{ u,v };
}


DEVICE Float3 sampleEnvironment(Float3 dir)
{
	vec2f tc{ uvOnSphere(dir) };
	owl::vec4f const texColor{
		tex2D<float4>(optixLaunchParams.environmentMap, tc.x, tc.y) };
	return vec3f{ texColor };
}


DEVICE Float3 makeFloat3(float3 f)
{
	return Float3{ f.x, f.y, f.z };
}



DEVICE Float3 tracePath(owl::Ray& ray, PerRayData& prd)
{
	auto& LP = optixLaunchParams;

	// hold total sum of accumulated radiance
	Float3 L{ 0.0f };
	// hold the path throughput weight
	//	 (f * cos(theta)) / pdf
	// => current implementation has f and cos already combined
	Float3 beta{ 1.0f };

	InterfaceStruct is;
	prd.is = &is;

	for (Int depth{ 0 }; depth < LP.maxDepth; ++depth)
	{
		/* FIND INTERSECTION */
		owl::traceRay(LP.world, ray, prd);


		/* TERMINATE PATH AND SAMPLE ENVIRONMENT*/
		if (prd.scatterEvent == ScatterEvent::MISS)
		{
			Float3 li{ 0.0f };
			if (!LP.useEnvironmentMap)
				li = 0.0f;
			else if (optixLaunchParams.environmentMap)
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
			LightStruct light{};
			GET(light, LightStruct, LP.lights, is.lightId);
			Float3 emission{ light.color * light.intensity };
			L += emission * beta;
			break;
		}


		/* SAMPLE BRDF OR PHASE FUNCTION */
		Float3 L{ 0.0f };
		toLocal(T, B, N, V);

		{
			MaterialStruct material{};
			if (is.matId >= 0) GET(material, MaterialStruct, LP.materials, is.matId);


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
		if (LP.lights.count != 0) 
		{
			// MAY BE INTRODUCE DOME SAMPLING LATER
			Int randMax{ LP.lights.count };
			Int randId{ (Int)min(prd.random() * randMax, randMax - 1.0f) };


			LightStruct lights{};



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


OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
	RayGenData const& self{ getProgramData<RayGenData>() };
	vec2i const pixelID{ getLaunchIndex() };
	Float3 color{ 0.0f };

	Random pxRand{ (Uint)pixelID.x, (Uint)pixelID.y };

	PerRayData prd{ pxRand };
	prd.scatterEvent = ScatterEvent::NONE;

	for (int32_t s{ 0 }; s < optixLaunchParams.samplesPerPixel; ++s)
	{
		// shot ray with slight randomness to make soft edges
		Float2 const rand{ prd.random(), prd.random() };
		Float2 const screen{ (Float2{pixelID} + rand) / Float2{self.fbSize} };

		// determine initial ray form the camera
		owl::Ray ray{ self.camera.origin, normalize(self.camera.llc
			+ screen.u * self.camera.horizontal
			+ screen.v * self.camera.vertical
			- self.camera.origin), T_MIN, T_MAX };

		color += tracePath(ray, prd);
	}

	// take the average of all samples per pixel and apply gamma correction
	color *= 1.0f / optixLaunchParams.samplesPerPixel;
	color = owl::sqrt(color);
	color = saturate<Float3>(color);


	// save result into the buffer
	const int fbOfs = pixelID.x + self.fbSize.x * (self.fbSize.y - 1 - pixelID.y);
	self.fbPtr[fbOfs]
		= owl::make_rgba(color);
}


OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
	PerRayData& prd{ getPRD<PerRayData>() };

	prd.is->t = optixGetRayTmax();

	// barycentrics
	float b1{ optixGetTriangleBarycentrics().x };
	float b2{ optixGetTriangleBarycentrics().y };
	float b0{ 1 - b1 - b2 };

	prd.is->UV = { b1, b2 };

	// get direction
	Float3 const direction{ makeFloat3(optixGetWorldRayDirection()) };

	prd.is->V = -direction;

	// get geometric data:
	TrianglesGeomData const& self = getProgramData<TrianglesGeomData>();
	uint32_t const primID{ optixGetPrimitiveIndex() };
	vec3i const index{ self.index[primID] };

	prd.is->matId = self.matId;
	prd.is->lightId = self.lightId;
	prd.is->prim = primID;

	// vertices for P and Ng
	Float3 const& p0{ self.vertex[index.x] };
	Float3 const& p1{ self.vertex[index.y] };
	Float3 const& p2{ self.vertex[index.z] };

	prd.is->TRI[0] = Float3{ p0 };
	prd.is->TRI[1] = Float3{ p1 };
	prd.is->TRI[2] = Float3{ p2 };

	prd.is->Ng = normalize(cross(p1 - p0, p2 - p0));
	prd.is->P = p0 * b0 + p1 * b1 + p2 * b2;

	// vertex normals for N
	Float3 const& n0{ self.normal[index.x] };
	Float3 const& n1{ self.normal[index.y] };
	Float3 const& n2{ self.normal[index.z] };

	prd.is->N = normalize(n0 * b0 + n1 * b1 + n2 * b2);

	// scatter event type
	prd.scatterEvent = ScatterEvent::BOUNCED;
}


OPTIX_MISS_PROGRAM(miss)()
{
	PerRayData& prd{ getPRD<PerRayData>() };
	prd.scatterEvent = ScatterEvent::MISS;
}
