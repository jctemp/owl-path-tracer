#include "device.hpp"
#include "PathTracer.hpp"
#include "Miss.hpp"
#include "ClosestHit.hpp"


__constant__ launch_params_data optixLaunchParams;

OPTIX_RAYGEN_PROGRAM(rayGenenration)()
{
	ray_gen_data const& self{ owl::getProgramData<ray_gen_data>() };
	ivec2 const pixelId{ owl::getLaunchIndex() };
	Random pxRand{ (uint32_t)pixelId.x, (uint32_t)pixelId.y };

	vec3 color{ 0.0f };
	for (int32_t s{ 0 }; s < optixLaunchParams.max_samples; ++s)
	{
		// shot ray with slight randomness to make soft edges
		vec2 const rand{ pxRand(), pxRand() };
		vec2 const screen{ (vec2{pixelId} + rand) / vec2{self.fbSize} };

		// determine initial ray form the camera
		owl::Ray ray{
			self.camera.origin,
			normalize(
				self.camera.llc + screen.u * self.camera.horizontal + screen.v * self.camera.vertical - self.camera.origin),
			T_MIN, T_MAX };

		color += tracePath(ray, pxRand);
	}

	// take the average of all samples per pixel and apply gamma correction
	color *= 1.0f / static_cast<float>(optixLaunchParams.max_samples);
	color = owl::sqrt(color);
	color = saturate<vec3>(color);


	// save result into the buffer
	const int fbOfs = pixelId.x + self.fbSize.x * (self.fbSize.y - 1 - pixelId.y);
	self.fbPtr[fbOfs]
		= owl::make_rgba(color);
}

