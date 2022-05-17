
#include <host/Renderer.hpp>
#include <pt/Types.hpp>
#include <SimpleLogger.hpp>
#include <device/device.hpp>

#include "utils/image_buffer.hpp"
#include "utils/mesh_loader.hpp"

using namespace owl;

Renderer renderer{};

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
	LightStruct light{};
	light.intensity = 10;

	std::vector<std::tuple<std::string, LightStruct>> li{
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

	renderer.frameBuffer = owlHostPinnedBufferCreate(
		renderer.context, OWL_INT, framebuffer_size.x * framebuffer_size.y);
	renderer.useEnvironmentMap = true;
	renderer.samplesPerPixel = 1024;
	renderer.maxDepth = 128;

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

	std::vector<LightStruct> lights{};
	for (auto& e : li)
		lights.push_back(std::get<LightStruct>(e));


	render(cam, materials, lights);

	// copy image buffer

	image_buffer result{ framebuffer_size.x, framebuffer_size.y,
		(Uint*)owlBufferGetPointer(renderer.frameBuffer, 0), image_buffer::tag::referenced };
	write_image(result, fmt::format("{}{}.png", prefix_path, "image"));

	release();
}
