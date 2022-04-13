#include <device/PathTracer.hpp>

#include <pathTracer/ObjLoader.hpp>
#include <pathTracer/StbUtils.hpp>

#include <owl/common/math/LinearSpace.h>
#include <owl/common/math/vec.h>
#include <owl/owl.h>

#include <map>
#include <vector>
#include <cuda_runtime.h>
#include <simpleLogger.hpp>

using namespace owl;

static ba::Renderer renderer{};
owl::vec2i const fbSize{ 1920, 1080 };
extern "C" char PathTracer_ptx[];

/// <summary>
/// Initialises the renderer with all necessary values to be ready.
/// Renderer is ready to receive data and render it after success.
/// </summary>
/// <returns>0 in case of success otherwise different</returns>
void init(void)
{
	renderer.context = owlContextCreate(nullptr, 1);
	renderer.module = owlModuleCreate(renderer.context, PathTracer_ptx);

	renderer.frameBuffer =
		owlHostPinnedBufferCreate(renderer.context, OWL_INT, fbSize.x * fbSize.y);

	OWLVarDecl launchParamsVars[]
	{
		{ "maxDepth",        OWL_USER_TYPE(uint32_t), OWL_OFFSETOF(LaunchParams, maxDepth)},
		{ "samplesPerPixel", OWL_USER_TYPE(uint32_t), OWL_OFFSETOF(LaunchParams, samplesPerPixel)},
		{ "world",           OWL_GROUP,               OWL_OFFSETOF(LaunchParams, world)},
		{ "environmentMap",  OWL_TEXTURE,             OWL_OFFSETOF(LaunchParams, environmentMap)},
		{ nullptr }
	};

	renderer.launchParams = 
		owlParamsCreate(renderer.context, sizeof(LaunchParams), launchParamsVars, -1);

	OWLVarDecl missProgVars[]
	{
		{ nullptr }
	};

	renderer.missProg =
		owlMissProgCreate(renderer.context, renderer.module, "miss", sizeof(MissProgData), missProgVars, -1);

	OWLVarDecl rayGenVars[]
	{
		{ "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData, fbPtr)},
		{ "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData, fbSize)},
		{ "camera.pos",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.pos)},
		{ "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_00)},
		{ "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_du)},
		{ "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_dv)},
		{ nullptr }
	};

	renderer.rayGen =
		owlRayGenCreate(renderer.context, renderer.module, "simpleRayGen", sizeof(RayGenData), rayGenVars, -1);

	OWLVarDecl trianglesGeomVars[]
	{
		{ "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, index) },
		{ "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, vertex) },
		{ "normal", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, normal) },
		{ "color",  OWL_FLOAT3, OWL_OFFSETOF(TrianglesGeomData, color) },
		{ nullptr }
	};

	renderer.trianglesGeomType = owlGeomTypeCreate(renderer.context, OWLGeomKind::OWL_GEOM_TRIANGLES,
		sizeof(TrianglesGeomData), trianglesGeomVars, -1);

	owlGeomTypeSetClosestHit(renderer.trianglesGeomType, 0, renderer.module, "TriangleMesh");

	owlBuildPrograms(renderer.context);
}

/// <summary>
/// Releases all resources of the renderer which are currently in
/// use.
/// </summary>
/// <returns>0 in case of success otherwise different</returns>
void release(void) 
{
	owlContextDestroy(renderer.context);
}

/// <summary>
/// </summary>
/// <returns></returns>
void setEnvironmentTexture(ba::ImageRgb const& texture)
{
	if (renderer.environmentMap != nullptr)
		owlTexture2DDestroy(renderer.environmentMap);

	renderer.environmentMap = owlTexture2DCreate(
		renderer.context,
		OWL_TEXEL_FORMAT_RGBA8,
		texture.width, texture.height,
		texture.texels.data(),
		OWL_TEXTURE_NEAREST,
		OWL_TEXTURE_CLAMP
	);
}

/// <summary>
/// Takes an intermediate form of a mesh and makes it ready for the
/// renderer. After loading successful the mesh can be rendered.
/// </summary>
/// <param name="m">An object of the type Mesh</param>
/// <returns>0 in case of success otherwise different</returns>
void add(ba::Mesh* m)
{
	ba::Mesh& mesh{ *m };

	auto& vertices{ mesh.vertex };
	auto& indices{ mesh.index };
	auto& normals{ mesh.normal };

	// set geometry in the buffers of the object
	OWLBuffer vertexBuffer{
		owlDeviceBufferCreate(renderer.context, OWL_FLOAT3, vertices.size(), vertices.data()) };

	OWLBuffer normalBuffer{
		owlDeviceBufferCreate(renderer.context, OWL_FLOAT3, normals.size(), normals.data()) };

	OWLBuffer indexBuffer{
		owlDeviceBufferCreate(renderer.context, OWL_INT3, indices.size(), indices.data()) };

	// prepare mesh for device
	OWLGeom geom{
		owlGeomCreate(renderer.context, renderer.trianglesGeomType) };

	// set specific vertex/index buffer => required for build the accel.
	owlTrianglesSetVertices(geom, vertexBuffer,
		vertices.size(), sizeof(owl::vec3f), 0);

	owlTrianglesSetIndices(geom, indexBuffer,
		indices.size(), sizeof(owl::vec3i), 0);

	// set sbt data
	owlGeomSetBuffer(geom, "vertex", vertexBuffer);
	owlGeomSetBuffer(geom, "normal", normalBuffer);
	owlGeomSetBuffer(geom, "index", indexBuffer);
	owlGeomSet3f(geom, "color", owl3f{ 0.5f, 1.0f, 0 });

	renderer.geoms.push_back(geom);
	renderer.requireBuild = true;
}

/// <summary>
/// Renderes the Meshes with the specifed render settings
/// </summary>
/// <returns>0 in case of success otherwise different</returns>
void render(ba::Camera const& cam)
{
	// 1) set mesh data into buffers
	if (renderer.geoms.size() > 0)
	{
		// Create Geom group and build world
		auto trianglesGroup =
			owlTrianglesGeomGroupCreate(renderer.context, renderer.geoms.size(), renderer.geoms.data());
		owlGroupBuildAccel(trianglesGroup);

		// Create an Instance group to make world
		renderer.world = owlInstanceGroupCreate(renderer.context, 1, &trianglesGroup);
		owlGroupBuildAccel(renderer.world);
	}
	else
	{
		renderer.world = owlInstanceGroupCreate(renderer.context, 0, nullptr);
		owlGroupBuildAccel(renderer.world);
	}

	// 2) set miss program data
	//owlMissProgSet3f(renderer.missProg, "name", value);

	// 3) calculate camera data
	float aspect{ fbSize.x / float(fbSize.y) };
	owl::vec3f camera_pos{ cam.lookFrom };
	owl::vec3f camera_d00{ owl::normalize(cam.lookAt - cam.lookFrom) };
	owl::vec3f camera_ddu{ cam.cosFovy * aspect * owl::normalize(owl::cross(camera_d00, cam.lookUp)) };
	owl::vec3f camera_ddv{ cam.cosFovy * owl::normalize(owl::cross(camera_ddu, camera_d00)) };
	camera_d00 -= 0.5f * camera_ddu;
	camera_d00 -= 0.5f * camera_ddv;

	// 4) set ray gen data
	owlRayGenSetBuffer(renderer.rayGen, "fbPtr", renderer.frameBuffer);
	owlRayGenSet2i(renderer.rayGen, "fbSize", (const owl2i&)fbSize);
	owlRayGenSet3f(renderer.rayGen, "camera.pos", (const owl3f&)camera_pos);
	owlRayGenSet3f(renderer.rayGen, "camera.dir_00", (const owl3f&)camera_d00);
	owlRayGenSet3f(renderer.rayGen, "camera.dir_du", (const owl3f&)camera_ddu);
	owlRayGenSet3f(renderer.rayGen, "camera.dir_dv", (const owl3f&)camera_ddv);

	// 5) set launch params
	owlParamsSetRaw(renderer.launchParams, "maxDepth", &renderer.maxDepth);
	owlParamsSetRaw(renderer.launchParams, "samplesPerPixel", &renderer.samplesPerPixel);
	owlParamsSetGroup(renderer.launchParams, "world", renderer.world);
	owlParamsSetTexture(renderer.launchParams, "environmentMap", renderer.environmentMap);

	// 6) build sbt tables and load data
	owlBuildPrograms(renderer.context);
	owlBuildPipeline(renderer.context);
	owlBuildSBT(renderer.context);

	// 7) compute image
	owlLaunch2D(renderer.rayGen, fbSize.x, fbSize.y, renderer.launchParams);
}

int main(void)
{
    std::vector<ba::Mesh*> meshes{ ba::loadOBJ("C:\\Users\\jamie\\Desktop\\sphere2.obj") };

    ba::Camera cam{ 
        {2.0f,1.0f,0.0f}, // look from
        {0.0f,0.5f,0.0f}, // look at
        {0.0f,1.0f,0.0f}, // look up
        0.88f			  // cosFov
    }; 

    ba::ImageRgb environmentTexture{};
    ba::loadImage(environmentTexture, "rooitou_park_4k.hdr", "C:/Users/jamie/Desktop");

	init();
	setEnvironmentTexture(environmentTexture);

    for (auto& m : meshes)
        add(m);

	renderer.samplesPerPixel = 512;
	renderer.maxDepth = 64;

    render(cam);

    ba::Image result{ fbSize.x, fbSize.y, (const uint32_t*)owlBufferGetPointer(renderer.frameBuffer, 0)};
    ba::writeImage(result, "image.png");

    release();
}
