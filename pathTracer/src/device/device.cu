#define GLM_FORCE_CUDA
#define PMEVENT( x ) asm volatile("pmevent " #x ";")

#include "device.hpp"
#include "core/core.hpp"

__constant__ launch_params_data optixLaunchParams;

OPTIX_RAYGEN_PROGRAM(rayGenenration)()
{
	RayGenData const& self{ owl::getProgramData<RayGenData>() };
	Int2 const pixelId{ owl::getLaunchIndex() };
	Random pxRand{ (uint32_t)pixelId.x, (uint32_t)pixelId.y };

	Float3 color{ 0.0f };
	for (int32_t s{ 0 }; s < optixLaunchParams.max_samples; ++s)
	{
		// shot ray with slight randomness to make soft edges
		Float2 const rand{ pxRand(), pxRand() };
		Float2 const screen{ (Float2{pixelId} + rand) / Float2{self.fbSize} };

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
	color = saturate<Float3>(color);


	// save result into the buffer
	const int fbOfs = pixelId.x + self.fbSize.x * (self.fbSize.y - 1 - pixelId.y);
	self.fbPtr[fbOfs]
		= owl::make_rgba(color);
}

