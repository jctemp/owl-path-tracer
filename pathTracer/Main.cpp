
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
	MaterialStruct mat1{ Material::DISNEY };
	mat1.baseColor = { .3f };
	mat1.metallic = 0.0f;
	mat1.specular = 0.0f;
	mat1.specularTint = 0.0f;
	mat1.roughness = 0.5f;
	mat1.sheen = 0.5f;
	mat1.sheenTint = 0.5f;
	mat1.clearcoat = 0.0f;
	mat1.clearcoatGloss = 0.0f;

	MaterialStruct mat2{ Material::DISNEY };
	mat2.baseColor = { .8f,.4f,.1f };
	mat2.metallic = 0.0f;
	mat2.specular = 0.0f;
	mat2.specularTint = 0.0f;
	mat2.roughness = 0.5f;
	mat2.sheen = 0.5f;
	mat2.sheenTint = 0.5f;
	mat2.clearcoat = 0.0f;
	mat2.clearcoatGloss = 0.0f;

	MaterialStruct test{ Material::DISNEY };
	test.baseColor = { .8f };
	test.subsurface = 0.0f;
	test.metallic = 1.0f;
	test.specular = 0.0f;
	test.specularTint = 0.0f;
	test.roughness = 0.05f;
	test.sheen = 0.0f;
	test.sheenTint = 0.5f;
	test.clearcoat = 0.0f;
	test.clearcoatGloss = 1.0f;

	std::vector<std::tuple<std::string, MaterialStruct>> mats{
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

	std::vector<MaterialStruct> materials{};
	for (auto& e : mats)
		materials.push_back(std::get<MaterialStruct>(e));

	std::vector<LightStruct> lights{};
	for (auto& e : li)
		lights.push_back(std::get<LightStruct>(e));


	render(cam, materials, lights);

	// copy image buffer

	image_buffer result{ framebuffer_size.x, framebuffer_size.y,
		(Uint *)owlBufferGetPointer(renderer.frameBuffer, 0), image_buffer::tag::referenced };
	write_image(result, fmt::format("{}{}.png", prefix_path, "image"));

	release();
}
