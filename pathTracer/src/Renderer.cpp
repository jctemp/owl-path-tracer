#include <pt/Types.hpp>
#include <pt/StbUtils.hpp>

#include <pt/Renderer.hpp>

#include <SimpleLogger.hpp>

extern "C" char PathTracer_ptx[];
extern Renderer renderer;

/// <summary>
/// Initialises the renderer with all necessary values to be ready.
/// Renderer is ready to receive data and render it after success.
/// </summary>
/// <returns>0 in case of success otherwise different</returns>
void init(void)
{
	renderer.context = owlContextCreate(nullptr, 1);
	renderer.module = owlModuleCreate(renderer.context, PathTracer_ptx);

	OWLVarDecl launchParamsVars[]
	{
		{ "maxDepth",          OWL_USER_TYPE(uint32_t), OWL_OFFSETOF(LaunchParams, maxDepth) },
		{ "samplesPerPixel",   OWL_USER_TYPE(uint32_t), OWL_OFFSETOF(LaunchParams, samplesPerPixel) },
		{ "materials",         OWL_BUFFER,              OWL_OFFSETOF(LaunchParams, materials)},
		{ "world",             OWL_GROUP,               OWL_OFFSETOF(LaunchParams, world) },
		{ "environmentMap",    OWL_TEXTURE,             OWL_OFFSETOF(LaunchParams, environmentMap) },
		{ "useEnvironmentMap", OWL_BOOL,                OWL_OFFSETOF(LaunchParams, useEnvironmentMap)},
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
		{ "fbPtr",             OWL_BUFPTR, OWL_OFFSETOF(RayGenData, fbPtr)},
		{ "fbSize",            OWL_INT2,   OWL_OFFSETOF(RayGenData, fbSize)},
		{ "camera.origin",     OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.origin)},
		{ "camera.llc",        OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.llc)},
		{ "camera.horizontal", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.horizontal)},
		{ "camera.vertical",   OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.vertical)},
		{ nullptr }
	};

	renderer.rayGen =
		owlRayGenCreate(renderer.context, renderer.module, "simpleRayGen", sizeof(RayGenData), rayGenVars, -1);

	OWLVarDecl trianglesGeomVars[]
	{
		{ "matId",  OWL_UINT,   OWL_OFFSETOF(TrianglesGeomData, matId) },
		{ "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, index) },
		{ "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, vertex) },
		{ "normal", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, normal) },
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
void setEnvironmentTexture(ImageRgb const& texture)
{
	if (renderer.environmentMap != nullptr)
		owlTexture2DDestroy(renderer.environmentMap);

	renderer.environmentMap = owlTexture2DCreate(
		renderer.context,
		OWL_TEXEL_FORMAT_RGBA8,
		texture.width, texture.height,
		texture.pixel,
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
void add(Mesh* m)
{
	Mesh& mesh{ *m };

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
	owlGeomSet1ui(geom, "matId", mesh.materialId);
	owlGeomSetBuffer(geom, "vertex", vertexBuffer);
	owlGeomSetBuffer(geom, "normal", normalBuffer);
	owlGeomSetBuffer(geom, "index", indexBuffer);

	renderer.geoms.push_back(geom);
	renderer.requireBuild = true;
}


/// <summary>
/// Renderes the Meshes with the specifed render settings
/// </summary>
/// <returns>0 in case of success otherwise different</returns>
void render(Camera const& cam, std::vector<Material> const& materials)
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
	// degrees * PI / 180.0f;
	Float aspect{ cam.fbSize.x / Float(cam.fbSize.y) };

	Float const theta{ cam.vfov * PI / 180.0f };
	Float const h{ tanf(theta / 2) };
	Float const viewportHeight{ 2.0f * h };
	Float const viewportWidth{ aspect * viewportHeight };

	Float3 const origin{ cam.lookFrom };
	Float3 const w{ normalize(cam.lookFrom - cam.lookAt) };
	Float3 const u{ normalize(cross(cam.lookUp, w)) };
	Float3 const v{ normalize(cross(w, u)) };

	Float3 const horizontal{ viewportWidth * u };
	Float3 const vertical{ viewportHeight * v };
	Float3 const llc{ origin - horizontal / 2.0f - vertical / 2.0f - w };

	// 4) set ray gen data
	owlRayGenSetBuffer(renderer.rayGen, "fbPtr", renderer.frameBuffer);
	owlRayGenSet2i(renderer.rayGen, "fbSize", (const owl2i&)cam.fbSize);
	owlRayGenSet3f(renderer.rayGen, "camera.origin", (const owl3f&)origin);
	owlRayGenSet3f(renderer.rayGen, "camera.llc", (const owl3f&)llc);
	owlRayGenSet3f(renderer.rayGen, "camera.horizontal", (const owl3f&)horizontal);
	owlRayGenSet3f(renderer.rayGen, "camera.vertical", (const owl3f&)vertical);

	// 5) set launch params
	auto materialBuffer{
		owlDeviceBufferCreate(renderer.context, OWL_USER_TYPE(Material), materials.size(), materials.data())
	};

	owlParamsSetRaw(renderer.launchParams, "maxDepth", &renderer.maxDepth);
	owlParamsSetRaw(renderer.launchParams, "samplesPerPixel", &renderer.samplesPerPixel);
	owlParamsSetBuffer(renderer.launchParams, "materials", materialBuffer);
	owlParamsSetGroup(renderer.launchParams, "world", renderer.world);
	owlParamsSetTexture(renderer.launchParams, "environmentMap", renderer.environmentMap);
	owlParamsSet1b(renderer.launchParams, "useEnvironmentMap", renderer.useEnvironmentMap);

	// 6) build sbt tables and load data
	owlBuildPrograms(renderer.context);
	owlBuildPipeline(renderer.context);
	owlBuildSBT(renderer.context);

	// 7) compute image
	SL_WARN("LAUNCHING TRACER");
	owlLaunch2D(renderer.rayGen, cam.fbSize.x, cam.fbSize.y, renderer.launchParams);
}
