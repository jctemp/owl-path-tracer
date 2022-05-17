
#include <pt/Types.hpp>
#include <device/device.hpp>

#include <device/camera.hpp>
#include <utils/image_buffer.hpp>
#include <utils/mesh_loader.hpp>

#include <fmt/core.h>
#include <fmt/color.h>

using namespace owl;


extern "C" char device_ptx[];

void optix_init()
{
	od.context = owlContextCreate(nullptr, 1);
	od.module = owlModuleCreate(od.context, device_ptx);
}

void optix_raygen_program()
{
	OWLVarDecl rayGenVars[]
	{
		{ "fbPtr",             OWL_BUFPTR, OWL_OFFSETOF(ray_gen_data, fbPtr)},
		{ "fbSize",            OWL_INT2,   OWL_OFFSETOF(ray_gen_data, fbSize)},
		{ "camera.origin",     OWL_FLOAT3, OWL_OFFSETOF(ray_gen_data, camera.origin)},
		{ "camera.llc",        OWL_FLOAT3, OWL_OFFSETOF(ray_gen_data, camera.llc)},
		{ "camera.horizontal", OWL_FLOAT3, OWL_OFFSETOF(ray_gen_data, camera.horizontal)},
		{ "camera.vertical",   OWL_FLOAT3, OWL_OFFSETOF(ray_gen_data, camera.vertical)},
		{ nullptr }
	};

	od.ray_gen_program =
		owlRayGenCreate(od.context, od.module, "rayGenenration", sizeof(ray_gen_data), rayGenVars, -1);
}

void optix_miss_program()
{

	OWLVarDecl missProgVars[]
	{
		{ nullptr }
	};

	od.miss_program =
		owlMissProgCreate(od.context, od.module, "miss", sizeof(miss_data), missProgVars, -1);
}

void optix_launch_params()
{
	OWLVarDecl launchParamsVars[]
	{
		{ "max_path_depth",          OWL_USER_TYPE(uint32_t), OWL_OFFSETOF(launch_params_data, max_path_depth) },
		{ "max_samples",   OWL_USER_TYPE(uint32_t), OWL_OFFSETOF(launch_params_data, max_samples) },
		{ "material_buffer",         OWL_BUFFER,              OWL_OFFSETOF(launch_params_data, material_buffer)},
		{ "light_buffer",            OWL_BUFFER,              OWL_OFFSETOF(launch_params_data, light_buffer)},
		{ "world",             OWL_GROUP,               OWL_OFFSETOF(launch_params_data, world) },
		{ "environment_map",    OWL_TEXTURE,             OWL_OFFSETOF(launch_params_data, environment_map) },
		{ "use_environment_map", OWL_BOOL,                OWL_OFFSETOF(launch_params_data, use_environment_map)},
		{ nullptr }
	};

	od.launch_params =
		owlParamsCreate(od.context, sizeof(launch_params_data), launchParamsVars, -1);
}

void optix_triangle_geom()
{
	OWLVarDecl trianglesGeomVars[]
	{
		{ "matId",   OWL_INT,   OWL_OFFSETOF(triangle_geom_data, matId) },
		{ "lightId", OWL_INT,   OWL_OFFSETOF(triangle_geom_data, lightId) },
		{ "index",   OWL_BUFPTR, OWL_OFFSETOF(triangle_geom_data, index) },
		{ "vertex",  OWL_BUFPTR, OWL_OFFSETOF(triangle_geom_data, vertex) },
		{ "normal",  OWL_BUFPTR, OWL_OFFSETOF(triangle_geom_data, normal) },
		{ nullptr }
	};

	od.triangle_geom = owlGeomTypeCreate(od.context, OWLGeomKind::OWL_GEOM_TRIANGLES,
		sizeof(triangle_geom_data), trianglesGeomVars, -1);

	owlGeomTypeSetClosestHit(od.triangle_geom, 0, od.module, "TriangleMesh");
}

void optix_destroy(void)
{
	owlContextDestroy(od.context);
}

void optix_set_environment_map(image_buffer const& texture)
{
	if (od.environment_map != nullptr)
		owlTexture2DDestroy(od.environment_map);

	od.environment_map = owlTexture2DCreate(
		od.context,
		OWL_TEXEL_FORMAT_RGBA8,
		texture.width, texture.height,
		texture.buffer,
		OWL_TEXTURE_NEAREST,
		OWL_TEXTURE_CLAMP
	);
}

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
		owlGeomCreate(od.context, od.triangle_geom) };

	// set specific vertex/index buffer => required for build the accel.
	owlTrianglesSetVertices(geom, vertexBuffer,
		vertices.size(), sizeof(owl::vec3f), 0);

	owlTrianglesSetIndices(geom, indexBuffer,
		indices.size(), sizeof(ivec3), 0);

	// set sbt data
	owlGeomSet1i(geom, "matId", e.materialId);
	owlGeomSet1i(geom, "lightId", e.lightId);
	owlGeomSetBuffer(geom, "vertex", vertexBuffer);
	owlGeomSetBuffer(geom, "normal", normalBuffer);
	owlGeomSetBuffer(geom, "index", indexBuffer);

	od.geoms.push_back(geom);
}

void render(camera_data const& camera, std::vector<material_data> const& materials, std::vector<light_data> const& lights)
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


	// 4) set ray gen data
	owlRayGenSetBuffer(od.ray_gen_program, "fbPtr", od.frame_buffer);
	owlRayGenSet2i(od.ray_gen_program, "fbSize", (const owl2i&)od.buffer_size);
	owlRayGenSet3f(od.ray_gen_program, "camera.origin", (const owl3f&)camera.origin);
	owlRayGenSet3f(od.ray_gen_program, "camera.llc", (const owl3f&)camera.llc);
	owlRayGenSet3f(od.ray_gen_program, "camera.horizontal", (const owl3f&)camera.horizontal);
	owlRayGenSet3f(od.ray_gen_program, "camera.vertical", (const owl3f&)camera.vertical);

	// 5) set launch params
	auto material_buffer{
		owlDeviceBufferCreate(od.context, OWL_USER_TYPE(material_data), materials.size(), materials.data())
	};

	auto light_buffer{
		owlDeviceBufferCreate(od.context, OWL_USER_TYPE(light_data), lights.size(), lights.data())
	};

	owlParamsSetRaw(od.launch_params, "max_path_depth", &od.max_path_depth);
	owlParamsSetRaw(od.launch_params, "max_samples", &od.max_samples);
	owlParamsSetBuffer(od.launch_params, "material_buffer", material_buffer);
	owlParamsSetBuffer(od.launch_params, "light_buffer", light_buffer);
	owlParamsSetGroup(od.launch_params, "world", od.world);
	owlParamsSetTexture(od.launch_params, "environment_map", od.environment_map);
	owlParamsSet1b(od.launch_params, "use_environment_map", od.use_environment_map);

	// 6) build sbt tables and load data
	owlBuildPrograms(od.context);
	owlBuildPipeline(od.context);
	owlBuildSBT(od.context);

	// 7) compute image
	fmt::print(fg(color::start), "LAUNCH TRACER\n");
	owlLaunch2D(od.ray_gen_program, od.buffer_size.x, od.buffer_size.y, od.launch_params);
	fmt::print(fg(color::stop), "LAUNCHING TRACER\n");
}

int main()
{
	auto const prefix_path{ std::string{"../../../.."} };
	auto const meshes{
		load_obj(fmt::format("{}/{}", prefix_path, "scenes/dragon.obj")) };

	camera cam{
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

	fmt::print(fg(color::log), "> MATERIALS\n");
	for (uint32_t i{ 0 }; i < mats.size(); i++)
		fmt::print("{} [{}]\n", std::get<std::string>(mats[i]), i);

	for (auto& [name, mesh] : meshes)
	{
		fmt::print("{}", name);
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

	fmt::print(fg(color::log), "> LIGHTS\n");
	for (uint32_t i{ 0 }; i < li.size(); i++)
		fmt::print("{} [{}]\n", std::get<std::string>(li[i]), i);


	for (auto& [name, mesh] : meshes)
	{
		fmt::print("{}", name);
		std::string in;
		std::getline(std::cin, in);
		if (!in.empty())
			entities[entities.size() - 1].lightId = std::stoi(in);
	}

	// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


	optix_init();
	optix_raygen_program();
	optix_miss_program();
	optix_launch_params();
	optix_triangle_geom();

	/* SCENE SELECT */

	od.buffer_size = ivec2{ 1024 };
	od.frame_buffer = owlHostPinnedBufferCreate(
		od.context, OWL_INT, od.buffer_size.x * od.buffer_size.y);
	od.use_environment_map = true;
	od.max_samples = 256;
	od.max_path_depth = 64;

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


	render(to_camera_data(cam, od.buffer_size), materials, lights);

	// copy image buffer

	image_buffer result{ od.buffer_size.x, od.buffer_size.y,
		(uint32_t*)owlBufferGetPointer(od.frame_buffer, 0), image_buffer::tag::referenced };
	write_image(result, fmt::format("{}/{}.png", prefix_path, "image"));

	optix_destroy();
}
