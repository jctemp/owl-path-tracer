#include <owl/owl.h>
#include <simpleLogger.hpp>

#include "device/tracer.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

using namespace owl;

extern "C" char tracer_ptx[];

vec2i fbSize{ 600, 400 };

const int NUM_VERTICES{ 4 };
vec3f vertices[]
{	// x y z
	{ -0.5f, +0.5f, -1.0f },
	{ -0.5f, -0.5f, -1.0f },
	{ +0.5f, -0.5f, -1.0f },
	{ +0.5f, +0.5f, -1.0f }
};

const int NUM_INDICIES{ 2 };
vec3i indices[]
{
	{ 0,1,2 }, { 2,3,0 }
};

int main(int argc, char* argv[])
{
	LOG("create context with single CUDA device and module to hold ptx");
	OWLContext context{ owlContextCreate(nullptr, 1) };
	OWLModule cudaModule{ owlModuleCreate(context, tracer_ptx) };

	LOG("Create buffers to communicate with GPU");

	OWLBuffer indexBuffer{ owlDeviceBufferCreate(context, OWL_INT3, NUM_INDICIES, indices) };
	OWLBuffer vertexBuffer{ owlDeviceBufferCreate(context, OWL_FLOAT3, NUM_VERTICES, vertices) };
	OWLBuffer frameBuffer{ owlHostPinnedBufferCreate(context, OWL_INT, fbSize.x * fbSize.y) };

	LOG("make OWLVars for custom data");

	OWLVarDecl trianglesGeomVars[]
	{
		{"index", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, index)},
		{"vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, vertex)},
		{/* SENTINEL ENTRY */}
	};

	OWLGeomType trianglesGeomType{ 
		owlGeomTypeCreate(
			context,						 // current application context
			OWLGeomKind::OWL_GEOM_TRIANGLES, // owl type
			sizeof(TrianglesGeomData),		 // size of a data struct
			trianglesGeomVars,				 // make program aware of vars in triangle
			2)								 // number of variables
	};
	owlGeomTypeSetClosestHit(trianglesGeomType, 0, cudaModule, "triHit");

	OWLVarDecl rayGenVars[]
	{
		{"fbPtr", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, fbPtr)},
		{"fbSize", OWL_INT2, OWL_OFFSETOF(RayGenData, fbSize)},
		{"world", OWL_GROUP, OWL_OFFSETOF(RayGenData, world)},
		{"c_pos", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.pos)},
		{"c_dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_00)},
		{"c_dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_du)},
		{"c_dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_dv)},
		{/* SENTINEL ENTRY */}
	};

	OWLVarDecl missVars[]
	{
		{"color", OWL_FLOAT3, OWL_OFFSETOF(MissData, color)},
		{/* SENTINEL ENTRY */}
	};

	LOG("Build triangles geometry");

	// create geometry type
	OWLGeom triangleGeom{ owlGeomCreate(context, trianglesGeomType) };

	// bind buffer to trianglesGeom
	owlTrianglesSetIndices(triangleGeom, indexBuffer, NUM_INDICIES, sizeof(vec3i), 0);
	owlTrianglesSetVertices(triangleGeom, vertexBuffer, NUM_VERTICES, sizeof(vec3f), 0);
	
	// map trianglesGeom data to GPU struct
	owlGeomSetBuffer(triangleGeom, "index", indexBuffer);
	owlGeomSetBuffer(triangleGeom, "vertex", vertexBuffer);

	LOG("setup group for accel. structure and build it");
	
	OWLGroup triangleGroup{ owlTrianglesGeomGroupCreate(context, 1, &triangleGeom) };
	owlGroupBuildAccel(triangleGroup);
	OWLGroup world{ owlInstanceGroupCreate(context, 1, &triangleGroup) };
	owlGroupBuildAccel(world);

	LOG("Miss program");

	OWLMissProg missProgram{ owlMissProgCreate(
			context,		  // current application context
			cudaModule,		  // module which is related to the ptx
			"rayMiss",		  // name of the OPTIX_MISS_PROGRAM
			sizeof(MissData), // size of the data struct
			missVars,		  // make program aware of struct vars
			-1) };			  // order of rays ?
	
	owlMissProgSet3f(missProgram, "color", owl3f{ .8f,.8f ,.8f });

	LOG("RayGen program");

	OWLRayGen rayGenProgram{ owlRayGenCreate(
		context,			  // current application context
		cudaModule,			  // module which is related to ptx
		"rayGen",			  // name of the OPTIX_RAYGEN_PROGRAM
		sizeof(RayGenData),	  // size of the data struct
		rayGenVars,			  // make program aware of struct vars
		-1) };				  // order of rays ?

	const vec2i fbSize(800, 600);
	const vec3f lookFrom(0.0f, 0.0f, 5.0f);
	const vec3f lookAt(0.0f, 0.0f, -1.0f);
	const vec3f lookUp(0.0f, 1.0f, 0.0f);
	const float cosFovy{ 0.66f };

	vec3f camera_pos = lookFrom;
	vec3f camera_d00
		= normalize(lookAt - lookFrom);
	float aspect = fbSize.x / float(fbSize.y);
	vec3f camera_ddu
		= cosFovy * aspect * normalize(cross(camera_d00, lookUp));
	vec3f camera_ddv
		= cosFovy * normalize(cross(camera_ddu, camera_d00));
	camera_d00 -= 0.5f * camera_ddu;
	camera_d00 -= 0.5f * camera_ddv;

	owlRayGenSetBuffer(rayGenProgram, "fbPtr", frameBuffer);
	owlRayGenSet2i(rayGenProgram, "fbSize", (const owl2i&) fbSize);
	owlRayGenSetGroup(rayGenProgram, "world", world);
	owlRayGenSet3f(rayGenProgram, "c_pos", (const owl3f&)camera_pos);
	owlRayGenSet3f(rayGenProgram, "c_dir_00", (const owl3f&)camera_d00);
	owlRayGenSet3f(rayGenProgram, "c_dir_du", (const owl3f&)camera_ddu);
	owlRayGenSet3f(rayGenProgram, "c_dir_dv", (const owl3f&)camera_ddv);

	LOG("BUILDING SBT TO TRACE GROUPS");


	LOG("Build Programs, Pipeline and SBT");

	owlBuildPrograms(context);
	owlBuildPipeline(context);
	owlBuildSBT(context);

	OK("Building successful...");


	LOG("Launch rayGen");

	owlRayGenLaunch2D(rayGenProgram, fbSize.x, fbSize.y);

	OK("Done rayGen launch");


	LOG("Writing picture");

	const uint32_t* fb{ (const uint32_t*)owlBufferGetPointer(frameBuffer, 0) };
	assert(fb);
	stbi_write_png("simpleTracer.png", fbSize.x, fbSize.y, 4, fb, fbSize.x * sizeof(uint32_t));
	
	OK("Written image to file");


	LOG("clean up ...");
	owlModuleRelease(cudaModule);
	owlContextDestroy(context);
}