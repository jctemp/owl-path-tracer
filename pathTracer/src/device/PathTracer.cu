#include "Globals.hpp"
#include "materials/Lambert.hpp"
#include "materials/Disney.hpp"
#include "lights/Light.hpp"

#include <owl/owl_device.h>
#include <optix_device.h>

#include "core/PathTracer.hpp"

using namespace owl;

__constant__ LaunchParams optixLaunchParams;

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

#include "./core/ClosestHit.hpp"
#include "./core/Miss.hpp"