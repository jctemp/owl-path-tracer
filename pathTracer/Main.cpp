#include <pt/ObjLoader.hpp>
#include <pt/StbUtils.hpp>
#include <pt/Types.hpp>
#include <pt/Renderer.hpp>

#include <SimpleLogger.hpp>

#include <map>
#include <vector>

#include <device/Globals.hpp>

using namespace owl;

Renderer renderer{};

//std::tuple<std::vector<std::string>, std::vector<Material*>> loadMaterial();
	//auto const [materialNames, materialData] {loadMaterial()};

int main(void)
{
	init();

	std::string prefixPath{ "../../../../" };

	//auto const [meshNames, meshData] {loadOBJ(prefixPath + "scenes/three-sphere-test.obj")};
	//Camera cam{
	//	{ 1024 },		  // image size
	//	{3.0f,0.5f,0.0f}, // look from
	//	{0.0f,0.5f,0.0f}, // look at
	//	{0.0f,1.0f,0.0f}, // look up
	//	60.0f			  // vfov
	//};

	//auto const [meshNames, meshData] {loadOBJ(prefixPath + "scenes/suzanne.obj")};
	//Camera cam{
	//	{ 600 },		  // image size
	//	{5.0f,5.0f,0.0f}, // look from
	//	{0.0f,0.75f,0.0f}, // look at
	//	{0.0f,1.0f,0.0f}, // look up
	//	30.0f			  // vfov
	//};

	//auto const [meshNames, meshData] {loadOBJ(prefixPath + "scenes/mitsuba.obj")};
	//Camera cam{
	//	{ 1024 },		  // image size
	//	{5.0f,3.0f,0.0f}, // look from
	//	{0.0f,0.75f,0.0f}, // look at
	//	{0.0f,1.0f,0.0f}, // look up
	//	30.0f			  // vfov
	//};

	//auto const [meshNames, meshData] {loadOBJ(prefixPath + "scenes/cornell-box-w-box-sphere.obj")};
	//auto const [meshNames, meshData] {loadOBJ(prefixPath + "scenes/cornell-box-w-boxes.obj")}; 
	//Camera cam{
	//	{ 600 },		  // image size
	//	{3.3f,1.0f,0.0f}, // look from
	//	{0.0f,1.0f,0.0f}, // look at
	//	{0.0f,1.0f,0.0f}, // look up
	//	45.0f			  // vfov
	//};

	auto const [meshNames, meshData] {loadOBJ(prefixPath + "scenes/dragon.obj")};
	Camera cam{
		{ 1024 },		  // image size
		{2.0f,1.2f,0.0f}, // look from
		{0.0f,0.5f,0.0f}, // look at
		{0.0f,1.0f,0.0f}, // look up
		50.0f			  // vfov
	};


	renderer.frameBuffer = owlHostPinnedBufferCreate(
		renderer.context, OWL_INT, cam.fbSize.x * cam.fbSize.y);
	renderer.useEnvironmentMap = false;
	renderer.samplesPerPixel = 1024;
	renderer.maxDepth = 128;

	//ImageRgb environmentTexture{};
	//loadImage(environmentTexture, "env.hdr", "C:/Users/jamie/Desktop");
	//setEnvironmentTexture(environmentTexture);

	/* BSDF */
	MaterialStruct mat1{ Material::DISNEY_BRDF };
	mat1.baseColor = { .3f };
	mat1.metallic = 0.0f;
	mat1.specular = 0.0f;
	mat1.specularTint = 0.0f;
	mat1.roughness = 0.5f;
	mat1.sheen = 0.5f;
	mat1.sheenTint = 0.5f;
	mat1.clearcoat = 0.0f;
	mat1.clearcoatGloss = 0.0f;


	MaterialStruct mat2{ Material::DISNEY_BRDF };
	mat2.baseColor = { .1f };
	mat2.metallic = 0.0f;
	mat2.specular = 0.0f;
	mat2.specularTint = 0.0f;
	mat2.roughness = 0.5f;
	mat2.sheen = 0.5f;
	mat2.sheenTint = 0.5f;
	mat2.clearcoat = 0.0f;
	mat2.clearcoatGloss = 0.0f;

	MaterialStruct test{ Material::DISNEY_BRDF };
	test.baseColor = { .8f };
	test.subsurface = 0.0f;
	test.metallic = 0.0f;
	test.specular = 0.0f;
	test.specularTint = 0.0f;
	test.roughness = 0.5f;
	test.sheen = 0.0f;
	test.sheenTint = 0.5f;
	test.clearcoat = 0.0f;
	test.clearcoatGloss = 1.0f;

	/*LightStruct light{};
	light.intensity = 5;*/

	std::vector<std::tuple<std::string, MaterialStruct>> mats{
		{"mat1", mat1},
		{"mat2", mat2},
		{"test", test},
	};

	std::vector<std::tuple<std::string, LightStruct>> li{
		//{"light", light}
	};

	SL_LOG("==== MATERIALS ===============================");
	for (uint32_t i{ 0 }; i < mats.size(); i++)
		fmt::print("{} [{}]\n", std::get<std::string>(mats[i]), i);

	
	for (uint32_t i{ 0 }; i < meshData.size(); i++)
	{
		fmt::print("{}: ", meshNames[i]);
		std::string in;
		std::getline(std::cin, in);
		if (!in.empty())
			meshData[i]->materialId = std::stoi(in);
	}

	SL_LOG("==== LIGHTS ===============================");
	for (uint32_t i{ 0 }; i < li.size(); i++)
		fmt::print("{} [{}]\n", std::get<std::string>(li[i]), i);

	for (uint32_t i{ 0 }; i < meshData.size(); i++)
	{
		fmt::print("{}: ", meshNames[i]);
		std::string in;
		std::getline(std::cin, in);
		if (!in.empty())
			meshData[i]->lightId = std::stoi(in);
		add(meshData[i]);
	}

	std::vector<MaterialStruct> materials{};
	for (auto& e : mats)
		materials.push_back(std::get<MaterialStruct>(e));

	std::vector<LightStruct> lights{};
	for (auto& e : li)
		lights.push_back(std::get<LightStruct>(e));

	render(cam, materials, lights);

	Image result{ cam.fbSize.x, cam.fbSize.y, (const uint32_t*)owlBufferGetPointer(renderer.frameBuffer, 0) };
	writeImage(result, fmt::format("{}{}.png", prefixPath, "image"));

	release();
}
