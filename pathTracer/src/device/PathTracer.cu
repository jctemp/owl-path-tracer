#include "Globals.hpp"
#include "materials/Lambert.hpp"
#include "materials/Microfacet.hpp"

#include <owl/owl_device.h>
#include <optix_device.h>

using namespace owl;

__constant__ LaunchParams optixLaunchParams;
//#define NORMAL

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



DEVICE Float3 tracePath(owl::Ray& ray, PerRayData &prd)
{
	auto& LP = optixLaunchParams;

	Float3 radiance{ 0.0f };
	Float3 pathThroughput{ 1.0f };

	ShadingData sd{ prd.random };
	ObjectData od{};
	Material md{};

	prd.od = &od;
	sd.md = &md;
	sd.od = &od;

	for (Int depth{ 0 }; depth < LP.maxDepth; ++depth)
	{
		/*
		 * 1) Shoot a ray through the scene.
		 *    => params: accel to trace against, the ray to trace, prd
		 */
		owl::traceRay(LP.world, ray, prd);


		/*
		 * 2) Check if we hit anything.
		 *    => If not we return the skybox color and break.
		 */
		if (prd.scatterEvent == ScatterEvent::MISS)
		{
			Float3 color{ 0.0f };
			if (!LP.useEnvironmentMap)
				color = 0.0f;
			else if (optixLaunchParams.environmentMap)
				color = sampleEnvironment(ray.direction);
			else
				color = mix(Float3{ 1.0f }, Float3{ 0.5f, 0.7f, 1.0f }, 0.5f * 
					(ray.direction.y + 1.0f));

			return color * pathThroughput;
		}


		/*
		 * 3) Load material data of prim and set required variables
		 *    => save to md
		 */
		GET(md, Material, LP.materials, od.matId);
		Float3& P{ od.P }, N{ od.N }, V{ od.V }, T{ sd.ts.T }, B{ sd.ts.B };
		onb(N, T, B);

		Float pdf{0.0f};
		Float3 L{ 0.0f };

		// move V to shading space
		toLocal(T, B, N, V);

		// sample direction
		auto f{Microfacet::sampleF(sd, V, L, pdf) };
		//auto f{ Microfacet::sampleF<Microfacet::Distributions::Blinn>(sd, V, L, pdf) };
		//auto f{ Microfacet::sampleF<Microfacet::Distributions::Beckmann>(sd, V, L, pdf) };

		 //end path if impossible
		//if (pdf < EPSILON)
		//	break;

		toWorld(T, B, N, L);

		pathThroughput *= f / (pdf);

		ray = owl::Ray{ P,L,T_MIN, T_MAX };
	}

	return { 0.0f };
}


OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
	RayGenData const& self{ getProgramData<RayGenData>() };
	vec2i const pixelID{ getLaunchIndex() };
	Float3 color{ 0.0f };

	PerRayData prd{ Random{}, nullptr , ScatterEvent::NONE };

	prd.random.init(pixelID.x, pixelID.y);

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

	prd.od->uv = { b1, b2 };

	// get direction
	Float3 const direction{ makeFloat3(optixGetWorldRayDirection()) };

	prd.od->V = -direction;

	// get geometric data:
	TrianglesGeomData const& self = getProgramData<TrianglesGeomData>();
	uint32_t const primID{ optixGetPrimitiveIndex() };
	vec3i const index{ self.index[primID] };

	prd.od->matId = self.matId;
	prd.od->prim = primID;

	// vertices for P and Ng
	Float3 const& p0{ self.vertex[index.x] };
	Float3 const& p1{ self.vertex[index.y] };
	Float3 const& p2{ self.vertex[index.z] };

	prd.od->Ng = normalize(cross(p1 - p0, p2 - p0));
	prd.od->P = p0 * b0 + p1 * b1 + p2 * b2;

	// vertex normals for N
	Float3 const& n0{ self.normal[index.x] };
	Float3 const& n1{ self.normal[index.y] };
	Float3 const& n2{ self.normal[index.z] };

	prd.od->N = normalize(n0 * b0 + n1 * b1 + n2 * b2);

	// scatter event type
	prd.scatterEvent = ScatterEvent::BOUNCED;
}


OPTIX_MISS_PROGRAM(miss)()
{
	PerRayData& prd{ getPRD<PerRayData>() };
	prd.scatterEvent = ScatterEvent::MISS;
}
