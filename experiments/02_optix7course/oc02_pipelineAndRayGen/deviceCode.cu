#include "LaunchParams.hpp"

#include <owl/owl.h>
#include <owl/owl_device.h>
#include <optix_device.h>

using namespace owl;

extern "C" __constant__ LaunchParams optixLaunchParams;

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
	vec2i pixelID{ getLaunchIndex() };
	if (pixelID == vec2i{ 0,0 })
		printf("#RAYGEN: Hello Opitx\n");

	const int r{ pixelID.x % 256 };
	const int g{ pixelID.y % 256 };
	const int b{ (pixelID.x + pixelID.y) % 256};

	const uint32_t rgba{ 0xff000000 | (r << 0) | (g << 8) | (b << 16) };
	const uint32_t fbIndex{ pixelID.x + optixLaunchParams.fbSize.x * pixelID.y };
	optixLaunchParams.colorBuffer[fbIndex] = rgba;
}

OPTIX_MISS_PROGRAM(miss)()
{
	/***/
}
