#include "tracer.hpp"
#include <optix_device.h>
#include <owl/owl_device.h>

using namespace owl;

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
	const RayGenData& self{ owl::getProgramData<RayGenData>() };
	const vec2i& pixelID{ owl::getLaunchIndex() };

	// x and y are now between [0,1]
	const vec2f screen{ (vec2f(pixelID) + vec2f(.5f)) / vec2f(self.fbSize) };

	Ray ray;
	ray.origin = self.camera.pos;
	ray.direction = normalize(
		self.camera.dir_00 +
		screen.x * self.camera.dir_du +
		screen.y * self.camera.dir_dv
	);

	const vec2f col{ vec2f(pixelID) / vec2f(self.fbSize) };
	vec3f color{col.x, col.y, 1.0f};
	traceRay(self.world, ray, color);

	// frame buffer (0,0) is top left, iterate over x
	int32_t fbIndex{ pixelID.x + self.fbSize.x * pixelID.y };
	self.fbPtr[fbIndex] = make_rgba(color);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
	// is the PRD data set in tracRay
	vec3f& prd{ getPRD<vec3f>() };

	const TrianglesGeomData& self{ getProgramData<TrianglesGeomData>() };

	// compute normal:
	const int   primID = optixGetPrimitiveIndex();
	const vec3i index = self.index[primID];
	const vec3f& A = self.vertex[index.x];
	const vec3f& B = self.vertex[index.y];
	const vec3f& C = self.vertex[index.z];
	const vec3f Ng = normalize(cross(B - A, C - A));

	const float3 rayDirOpitx{ optixGetWorldRayDirection() };
	const vec3f rayDir{ rayDirOpitx.x, rayDirOpitx.y ,rayDirOpitx.z };
	prd = (.2f + .8f * fabs(dot(rayDir, Ng))) * self.color;
}

OPTIX_MISS_PROGRAM(miss)()
{
	const vec2i pixelID = owl::getLaunchIndex();
	const MissData& self = owl::getProgramData<MissData>();
	vec3f& prd = owl::getPRD<vec3f>();
	int pattern = (pixelID.x / 8) ^ (pixelID.y / 8);
	prd = (pattern & 1) ? self.color1 : self.color0;
}