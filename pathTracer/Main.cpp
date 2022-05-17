
#include <pt/Types.hpp>
#include <SimpleLogger.hpp>
#include <device/device.hpp>

#include "utils/image_buffer.hpp"
#include "utils/mesh_loader.hpp"

using namespace owl;


extern "C" char device_ptx[];

/// <summary>
/// Initialises the renderer with all necessary values to be ready.
/// Renderer is ready to receive data and render it after success.
/// </summary>
/// <returns>0 in case of success otherwise different</returns>
void init(void)
{
	od.context = owlContextCreate(nullptr, 1);
	od.module = owlModuleCreate(od.context, device_ptx);

	OWLVarDecl launchParamsVars[]
	{
		{ "maxDepth",          OWL_USER_TYPE(uint32_t), OWL_OFFSETOF(LaunchParams, maxDepth) },
		{ "samplesPerPixel",   OWL_USER_TYPE(uint32_t), OWL_OFFSETOF(LaunchParams, samplesPerPixel) },
		{ "materials",         OWL_BUFFER,              OWL_OFFSETOF(LaunchParams, materials)},
		{ "lights",            OWL_BUFFER,              OWL_OFFSETOF(LaunchParams, lights)},
		{ "world",             OWL_GROUP,               OWL_OFFSETOF(LaunchParams, world) },
		{ "environmentMap",    OWL_TEXTURE,             OWL_OFFSETOF(LaunchParams, environmentMap) },
		{ "useEnvironmentMap", OWL_BOOL,                OWL_OFFSETOF(LaunchParams, useEnvironmentMap)},
		{ nullptr }
	};

	od.launchParams =
		owlParamsCreate(od.context, sizeof(LaunchParams), launchParamsVars, -1);

	OWLVarDecl missProgVars[]
	{
		{ nullptr }
	};

	od.missProg =
		owlMissProgCreate(od.context, od.module, "miss", sizeof(MissProgData), missProgVars, -1);

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

	od.rayGenenration =
		owlRayGenCreate(od.context, od.module, "rayGenenration", sizeof(RayGenData), rayGenVars, -1);

	OWLVarDecl trianglesGeomVars[]
	{
		{ "matId",   OWL_INT,   OWL_OFFSETOF(TrianglesGeomData, matId) },
		{ "lightId", OWL_INT,   OWL_OFFSETOF(TrianglesGeomData, lightId) },
		{ "index",   OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, index) },
		{ "vertex",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, vertex) },
		{ "normal",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, normal) },
		{ nullptr }
	};

	od.trianglesGeomType = owlGeomTypeCreate(od.context, OWLGeomKind::OWL_GEOM_TRIANGLES,
		sizeof(TrianglesGeomData), trianglesGeomVars, -1);

	owlGeomTypeSetClosestHit(od.trianglesGeomType, 0, od.module, "TriangleMesh");

	owlBuildPrograms(od.context);
}


/// <summary>
/// Releases all resources of the renderer which are currently in
/// use.
/// </summary>
/// <returns>0 in case of success otherwise different</returns>
void release(void)
{
	owlContextDestroy(od.context);
}


/// <summary>
/// </summary>
/// <returns></returns>
void setEnvironmentTexture(image_buffer const& texture)
{
	if (od.environmentMap != nullptr)
		owlTexture2DDestroy(od.environmentMap);

	od.environmentMap = owlTexture2DCreate(
		od.context,
		OWL_TEXEL_FORMAT_RGBA8,
		texture.width, texture.height,
		texture.buffer,
		OWL_TEXTURE_NEAREST,
		OWL_TEXTURE_CLAMP
	);
}


/// <summary>
/// Takes an intermediate form of a mesh and makes it ready for the
/// od. After loading successful the mesh can be rendered.
/// </summary>
/// <param name="m">An object of the type Mesh</param>
/// <returns>0 in case of success otherwise different</returns>
void add(mesh* m, entity e)
{
	mesh& mesh{ *m };

	auto& vertices{ mesh.vertices };
	auto& indices{ mesh.indices };
	auto& normals{ mesh.normals };

	// set geometry in the buffers of the object
	OWLBuffer vertexBuffer{
		owlDeviceBufferCreate(od.context, OWL_FLOAT3, vertices.size(), vertices.data()) };

	OWLBuffer normalBuffer{
		owlDeviceBufferCreate(od.context, OWL_FLOAT3, normals.size(), normals.data()) };

	OWLBuffer indexBuffer{
		owlDeviceBufferCreate(od.context, OWL_INT3, indices.size(), indices.data()) };

	// prepare mesh for device
	OWLGeom geom{
		owlGeomCreate(od.context, od.trianglesGeomType) };

	// set specific vertex/index buffer => required for build the accel.
	owlTrianglesSetVertices(geom, vertexBuffer,
		vertices.size(), sizeof(owl::vec3f), 0);

	owlTrianglesSetIndices(geom, indexBuffer,
		indices.size(), sizeof(owl::vec3i), 0);

	// set sbt data
	owlGeomSet1i(geom, "matId", e.materialId);
	owlGeomSet1i(geom, "lightId", e.lightId);
	owlGeomSetBuffer(geom, "vertex", vertexBuffer);
	owlGeomSetBuffer(geom, "normal", normalBuffer);
	owlGeomSetBuffer(geom, "index", indexBuffer);

	od.geoms.push_back(geom);
	od.requireBuild = true;
}


/// <summary>
/// Renderes the Meshes with the specifed render settings
/// </summary>
/// <returns>0 in case of success otherwise different</returns>
void render(Camera const& cam, std::vector<material_data> const& materials, std::vector<light_data> const& lights)
{
	// 1) set mesh data into buffers
	if (od.geoms.size() > 0)
	{
		// Create Geom group and build world
		auto trianglesGroup =
			owlTrianglesGeomGroupCreate(od.context, od.geoms.size(), od.geoms.data());
		owlGroupBuildAccel(trianglesGroup);

		// Create an Instance group to make world
		od.world = owlInstanceGroupCreate(od.context, 1, &trianglesGroup);
		owlGroupBuildAccel(od.world);
	}
	else
	{
		od.world = owlInstanceGroupCreate(od.context, 0, nullptr);
		owlGroupBuildAccel(od.world);
	}

	// 2) set miss program data
	//owlMissProgSet3f(od.missProg, "name", value);

	// 3) calculate camera data
	// degrees * PI / 180.0f;
	Float aspect{ cam.fbSize.x / Float(cam.fbSize.y) };

	Float const theta{ cam.vfov * Float(M_PI) / 180.0f };
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
	owlRayGenSetBuffer(od.rayGenenration, "fbPtr", od.frameBuffer);
	owlRayGenSet2i(od.rayGenenration, "fbSize", (const owl2i&)cam.fbSize);
	owlRayGenSet3f(od.rayGenenration, "camera.origin", (const owl3f&)origin);
	owlRayGenSet3f(od.rayGenenration, "camera.llc", (const owl3f&)llc);
	owlRayGenSet3f(od.rayGenenration, "camera.horizontal", (const owl3f&)horizontal);
	owlRayGenSet3f(od.rayGenenration, "camera.vertical", (const owl3f&)vertical);

	// 5) set launch params
	auto materialBuffer{
		owlDeviceBufferCreate(od.context, OWL_USER_TYPE(material_data), materials.size(), materials.data())
	};
	owlParamsSetBuffer(od.launchParams, "materials", materialBuffer);

	auto lightBuffer{
		owlDeviceBufferCreate(od.context, OWL_USER_TYPE(light_data), lights.size(), lights.data())
	};
	owlParamsSetBuffer(od.launchParams, "lights", lightBuffer);

	owlParamsSetRaw(od.launchParams, "maxDepth", &od.maxDepth);
	owlParamsSetRaw(od.launchParams, "samplesPerPixel", &od.samplesPerPixel);
	owlParamsSetGroup(od.launchParams, "world", od.world);
	owlParamsSetTexture(od.launchParams, "environmentMap", od.environmentMap);
	owlParamsSet1b(od.launchParams, "useEnvironmentMap", od.useEnvironmentMap);

	// 6) build sbt tables and load data
	owlBuildPrograms(od.context);
	owlBuildPipeline(od.context);
	owlBuildSBT(od.context);

	// 7) compute image
	SL_WARN("LAUNCHING TRACER");
	owlLaunch2D(od.rayGenenration, cam.fbSize.x, cam.fbSize.y, od.launchParams);
}


int main(void)
{
	auto const prefix_path{ std::string{"../../../../"} };
	auto const meshes{ load_obj(prefix_path + "scenes/dragon.obj") };

	Camera cam{
	{ 1024 },		  // image size
	{3.0f,0.5f,0.0f}, // look from
	{0.0f,0.5f,0.0f}, // look at
	{0.0f,1.0f,0.0f}, // look up
	60.0f			  // vfov
	};

	/* BSDF */
	material_data mat1{
		material_data::type::DISNEY,
		{.3f, .3f, .3f},	  
		0.0f ,				  
		{ 1.0f, 0.2f, 0.1f }, 
		{ 0.8f, 0.8f, 0.8f }, 
		0.0f,				  
		0.0f,				  
		0.0f,				  
		0.5f,				  
		0.5f,				  
		0.5f,				  
		0.0f,				  
		0.0f,				  
		1.45f				  
	};

	material_data mat2{
		material_data::type::DISNEY,
		{.8f,.4f,.1f},
		0.0f ,
		{ 1.0f, 0.2f, 0.1f },
		{ 0.8f, 0.8f, 0.8f },
		0.0f,
		0.0f,
		0.0f,
		0.5f,
		0.5f,
		0.5f,
		0.0f,
		0.0f,
		1.45f
	};

	material_data test{
		material_data::type::DISNEY,
		{.8f, .8f, .8f},
		0.0f ,
		{ 1.0f, 0.2f, 0.1f },
		{ 0.8f, 0.8f, 0.8f },
		1.0f,
		0.0f,
		0.0f,
		0.05f,
		0.0f,
		0.5f,
		0.0f,
		0.0f,
		1.45f
	};

	std::vector<std::tuple<std::string, material_data>> mats{
		{"mat1", mat1},
		{"mat2", mat2},
		{"test", test},
	};

	std::vector<entity> entities{};

	SL_LOG("==== MATERIALS ===============================");
	for (uint32_t i{ 0 }; i < mats.size(); i++)
		fmt::print("{} [{}]\n", std::get<std::string>(mats[i]), i);

	for (auto& [name, mesh] : meshes)
	{
		fmt::print("{}: ", name);
		std::string in;
		std::getline(std::cin, in);
		if (!in.empty())
			entities.push_back({ std::stoi(in) });
	}

	/* LIGHTS */
	light_data light{};
	light.intensity = 10;

	std::vector<std::tuple<std::string, light_data>> li{
		{"light", light}
	};

	SL_LOG("==== LIGHTS ===============================");
	for (uint32_t i{ 0 }; i < li.size(); i++)
		fmt::print("{} [{}]\n", std::get<std::string>(li[i]), i);

	for (auto& [name, mesh] : meshes)
	{
		fmt::print("{}: ", name);
		std::string in;
		std::getline(std::cin, in);
		if (!in.empty())
			entities[entities.size() - 1].lightId = std::stoi(in);
	}

	// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

	init();

	/* SCENE SELECT */

	glm::ivec2 constexpr framebuffer_size{ 1024 };

	od.frameBuffer = owlHostPinnedBufferCreate(
		od.context, OWL_INT, framebuffer_size.x * framebuffer_size.y);
	od.useEnvironmentMap = true;
	od.samplesPerPixel = 1024;
	od.maxDepth = 128;

	/* ENVMAP */
	if (false)
	{
		image_buffer environmentTexture{};
		environmentTexture = load_image("env.hdr", "C:/Users/jamie/Desktop");
	}

	/* RENDER */
	uint64_t i{ 0 };
	for (auto& [n, m] : meshes)
		add(m.get(), entities[i++]);

	std::vector<material_data> materials{};
	for (auto& e : mats)
		materials.push_back(std::get<material_data>(e));

	std::vector<light_data> lights{};
	for (auto& e : li)
		lights.push_back(std::get<light_data>(e));


	render(cam, materials, lights);

	// copy image buffer

	image_buffer result{ framebuffer_size.x, framebuffer_size.y,
		(Uint*)owlBufferGetPointer(od.frameBuffer, 0), image_buffer::tag::referenced };
	write_image(result, fmt::format("{}{}.png", prefix_path, "image"));

	release();
}
