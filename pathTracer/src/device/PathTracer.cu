#include "Globals.hpp"
#include "materials/Lambert.hpp"
#include "materials/DisneyBrdf.hpp"

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

	Float3 radiance{ 0.0f };
	Float3 accumulatedRadiance{ 0.0f };
	Float3 pathThroughput{ 1.0f };

	InterfaceStruct is;

	prd.is = &is;

	for (Int depth{ 0 }; depth < LP.maxDepth; ++depth)
	{
		/* FIND INTERSECTION */
		owl::traceRay(LP.world, ray, prd);


		/* SAMPLE ENVIRONMENT FOR NO HIT*/
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
			//li = Float3{ 1.0f };

			return li * pathThroughput;
		}


		/* LOAD OBJECT DATA */
		MaterialStruct materials{};
		LightStruct lights{};

		if (is.matId >= 0)
		{
			GET(materials, MaterialStruct, LP.materials, is.matId);
		}

		if (is.lightId >= 0)
		{
			GET(lights, LightStruct, LP.lights, is.lightId);
		}

		Float3& P{ is.P }, N{ is.N }, V{ is.V }, Ng{ is.Ng };
		Float3 T{}, B{};

		/* HANDLE LIGHTS */
		if (materials.type == Material::EMISSION)
		{
			if (getLaunchIndex().x == 0)
				printf("%d\n", lights.type == Light::NONE);
			return materials.emission * pathThroughput;
		}

		onb(N, T, B);
		toLocal(T, B, N, V);


		/* SAMPLE MATERIAL */
		Float pdf{ 0.0f };
		Float3 bsdf{ 0.0f };
		Float3 L{ 0.0f };

		sampleDisneyBSDF(materials, V, L, prd.random, bsdf, pdf);

		if (isnan(bsdf.x) || isnan(bsdf.y) || isnan(bsdf.z))
			printf("bsdf %f, %f, %f is nan\n", bsdf.x, bsdf.y, bsdf.z);
		if (isinf(bsdf.x) || isinf(bsdf.y) || isinf(bsdf.z))
			printf("bsdf %f, %f, %f is inf\n", bsdf.x, bsdf.y, bsdf.z);

		// end path if impossible
		if (pdf <= 0.0f)
			break;

		// because of the LTE equation => f_d * L(p,\omega_i) * | cos\theta |
		// => the pathTroughput defines how much radiance is reaching the view 
		// after the material interaction
		pathThroughput *= bsdf / pdf;


		/* RUSSIAN ROULETTE */
		// at least 3 bounces required to avoid bias
		float pmax = max(pathThroughput.x, max(pathThroughput.y, pathThroughput.z));
		if (depth > 3 && prd.random() > pmax) {
			break;
		}

		toWorld(T, B, N, L);
		ray = owl::Ray{ P,L,T_MIN, T_MAX };
	}

	return { 0.0f };
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

	// barycentrics
	float b1{ optixGetTriangleBarycentrics().x };
	float b2{ optixGetTriangleBarycentrics().y };
	float b0{ 1 - b1 - b2 };

	prd.is->uv = { b1, b2 };

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
