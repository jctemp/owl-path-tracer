#ifndef CLOSEST_HIT_HPP
#define CLOSEST_HIT_HPP
#pragma once

#include "device.hpp"
#include "materials/Lambert.hpp"
#include "materials/Disney.hpp"

#include <owl/owl_device.h>
#include <optix_device.h>

using namespace owl;

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
	per_ray_data& prd{ getPRD<per_ray_data>() };

	prd.is->t = optixGetRayTmax();

	// barycentrics
	float b1{ optixGetTriangleBarycentrics().x };
	float b2{ optixGetTriangleBarycentrics().y };
	float b0{ 1 - b1 - b2 };

	prd.is->UV = { b1, b2 };

	// get direction
	auto ray_dir{ optixGetWorldRayDirection() };
	vec3 const direction{
		ray_dir.x,
		ray_dir.y,
		ray_dir.z
	};

	prd.is->V = -direction;

	// get geometric data:
	triangle_geom_data const& self = getProgramData<triangle_geom_data>();
	uint32_t const primID{ optixGetPrimitiveIndex() };
	ivec3 const index{ self.index[primID] };

	prd.is->matId = self.matId;
	prd.is->lightId = self.lightId;
	prd.is->prim = primID;

	// vertices for P and Ng
	vec3 const& p0{ self.vertex[index.x] };
	vec3 const& p1{ self.vertex[index.y] };
	vec3 const& p2{ self.vertex[index.z] };

	prd.is->TRI[0] = vec3{ p0 };
	prd.is->TRI[1] = vec3{ p1 };
	prd.is->TRI[2] = vec3{ p2 };

	prd.is->Ng = normalize(cross(p1 - p0, p2 - p0));
	prd.is->P = p0 * b0 + p1 * b1 + p2 * b2;

	// vertex normals for N
	vec3 const& n0{ self.normal[index.x] };
	vec3 const& n1{ self.normal[index.y] };
	vec3 const& n2{ self.normal[index.z] };

	prd.is->N = normalize(n0 * b0 + n1 * b1 + n2 * b2);

	// scatter event type
	prd.scatterEvent = ScatterEvent::BOUNCED;
}

#endif // !CLOSEST_HIT_HPP
