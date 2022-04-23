#include <pt/ObjLoader.hpp>
#include <pt/StbUtils.hpp>
#include <pt/Types.hpp>
#include <pt/Renderer.hpp>

#include <SimpleLogger.hpp>

#include <map>
#include <vector>

using namespace owl;

Renderer renderer{};

//std::tuple<std::vector<std::string>, std::vector<Material*>> loadMaterial();

int main(void)
{
	init();

	auto const [meshNames, meshData] {loadOBJ("../../../../scenes/cornell-box-w-boxes.obj")};
	//auto const [materialNames, materialData] {loadMaterial()};

	Camera cam{
		{ 600 },		  // image size
		{2.1f,1.0f,0.0f}, // look from
		{0.0f,1.0f,0.0f}, // look at
		{0.0f,1.0f,0.0f}, // look up
		80.0f			  // vfov
	};

	renderer.frameBuffer = owlHostPinnedBufferCreate(
		renderer.context, OWL_INT, cam.fbSize.x * cam.fbSize.y);
	renderer.useEnvironmentMap = true;
	renderer.samplesPerPixel = 124;
	renderer.maxDepth = 128;

	//ImageRgb environmentTexture{};
	//loadImage(environmentTexture, "env.hdr", "C:/Users/jamie/Desktop");
	//setEnvironmentTexture(environmentTexture);

	Material lambert_n{ Material::Type::LAMBERT, {0.8f, 0.8f, 0.8f}, 0.0f, { 0.0f } };
	Material lambert_c{ Material::Type::LAMBERT, {0.7f, 0.3f, 0.3f}, 0.0f, { 0.0f } };
	Material metal{ Material::Type::METAL,   {0.8f, 0.6f, 0.2f}, 0.0f, { 0.0f } };
	Material light{ Material::Type::LIGHT,   {},                 0.0f, { 10.0f } };

	std::vector<std::tuple<std::string, Material>> mats{};
	mats.emplace_back("lambert_n", lambert_n);
	mats.emplace_back("lambert_c", lambert_c);
	mats.emplace_back("metal", metal);
	mats.emplace_back("light", light);

	SL_LOG("PLEASE SELECT A MATERIAL FOR MESH");
	for (uint32_t i{ 0 }; i < mats.size(); i++)
		fmt::print("{} [{}]\n", std::get<std::string>(mats[i]), i);

	for (uint32_t i{ 0 }; i < meshData.size(); i++)
	{
		fmt::print("{}: ", meshNames[i]);
		std::cin >> meshData[i]->materialId;
		add(meshData[i]);
	}

	std::vector<Material> materials{};
	for (auto& e : mats)
		materials.push_back(std::get<Material>(e));

	render(cam, materials);

	Image result{ cam.fbSize.x, cam.fbSize.y, (const uint32_t*)owlBufferGetPointer(renderer.frameBuffer, 0) };
	writeImage(result, "../../../../image.png");

	release();
}

// 1. load materials
// 2. load obj
// 3. reference mesh to material
// 4. add mesh & material to renderer