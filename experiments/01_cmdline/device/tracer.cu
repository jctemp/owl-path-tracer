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

OPTIX_CLOSEST_HIT_PROGRAM(triHit)()
{
	// is the PRD data set in tracRay
	vec3f& prd{ getPRD<vec3f>() };
	prd = vec3f{ prd.x, prd.y, 0 };
}

OPTIX_MISS_PROGRAM(rayMiss)()
{
	// is the PRD data set in tracRay
	vec3f& prd{ getPRD<vec3f>() };
	const MissData& self = owl::getProgramData<MissData>();
	prd = self.color;
}