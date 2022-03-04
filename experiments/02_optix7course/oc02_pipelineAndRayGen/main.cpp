#include "LaunchParams.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <owl/owl.h>
#include <simpleLogger.hpp>
#include <stb_image_write.h>

using namespace owl;

extern "C" char deviceCode_ptx[];

const char* outFileName = "oc02_pipelineAndRayGen.png";
const vec2i fbSize(800, 600);

extern "C" int main(int argc, char* argv[])
{
	OWLContext context{ owlContextCreate(nullptr, 1) };
	OWLModule module{ owlModuleCreate(context, deviceCode_ptx) };

	OWLVarDecl launchParamVars[]
	{
		{ "frameID", OWL_INT, OWL_OFFSETOF(LaunchParams, frameID) },
		{ "colorBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, colorBuffer) },
		{ "fbSize", OWL_INT2, OWL_OFFSETOF(LaunchParams, fbSize) },
		{ /* SENTINEL */ nullptr }
	};

	OWLLaunchParams launchParams{ owlParamsCreate(context, sizeof(LaunchParams), launchParamVars, -1) };

	OWLRayGen rayGen{ owlRayGenCreate(context, module, "rayGen", /* no sbt data */0, nullptr, -1) };

	OWLBuffer frameBuffer = owlHostPinnedBufferCreate(context, OWL_INT, fbSize.x * fbSize.y);

	owlParamsSet1i(launchParams, "frameID", 0);
	owlParamsSetBuffer(launchParams, "colorBuffer", frameBuffer);
	owlParamsSet2i(launchParams, "fbSize", (owl2i const&)fbSize);

	owlBuildPrograms(context);
	owlBuildPipeline(context);
	owlBuildSBT(context);

	owlLaunch2D(rayGen, fbSize.x, fbSize.y, launchParams);

	const uint32_t* fb = (const uint32_t*)owlBufferGetPointer(frameBuffer, 0);
	stbi_write_png(outFileName, fbSize.x, fbSize.y, 4,
		fb, fbSize.x * sizeof(uint32_t));

	owlBufferRelease(frameBuffer);
	owlRayGenRelease(rayGen);
	owlModuleRelease(module);
	owlContextDestroy(context);
	OK("SUCCESS");
}
