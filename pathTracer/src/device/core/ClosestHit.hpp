#ifndef CLOSEST_HIT_HPP
#define CLOSEST_HIT_HPP
#pragma once

#include "Globals.hpp"
#include "materials/Lambert.hpp"
#include "materials/Disney.hpp"

#include <owl/owl_device.h>
#include <optix_device.h>

using namespace owl;

DEVICE Float3 makeFloat3(float3 f)
{
	return Float3{ f.x, f.y, f.z };
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

#endif // !CLOSEST_HIT_HPP
