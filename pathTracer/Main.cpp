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

	auto const [meshNames, meshData] {loadOBJ(prefixPath + "scenes/suzanne.obj")};
	Camera cam{
		{ 600 },		  // image size
		{5.0f,5.0f,0.0f}, // look from
		{0.0f,0.75f,0.0f}, // look at
		{0.0f,1.0f,0.0f}, // look up
		30.0f			  // vfov
	};

	//auto const [meshNames, meshData] {loadOBJ(prefixPath + "scenes/cornell-box-w-box-sphere.obj")};
	//auto const [meshNames, meshData] {loadOBJ(prefixPath + "scenes/cornell-box-w-boxes.obj")}; 
	//Camera cam{
	//	{ 600 },		  // image size
	//	{3.3f,1.0f,0.0f}, // look from
	//	{0.0f,1.0f,0.0f}, // look at
	//	{0.0f,1.0f,0.0f}, // look up
	//	45.0f			  // vfov
	//};

	//auto const [meshNames, meshData] {loadOBJ(prefixPath + "scenes/dragon.obj")};
	//Camera cam{
	//	{ 600 },		  // image size
	//	{2.0f,1.2f,0.0f}, // look from
	//	{0.0f,1.0f,0.0f}, // look at
	//	{0.0f,1.0f,0.0f}, // look up
	//	50.0f			  // vfov
	//};


	renderer.frameBuffer = owlHostPinnedBufferCreate(
		renderer.context, OWL_INT, cam.fbSize.x * cam.fbSize.y);
	renderer.useEnvironmentMap = true;
	renderer.samplesPerPixel = 1024;
	renderer.maxDepth = 128;

	//ImageRgb environmentTexture{};
	//loadImage(environmentTexture, "env.hdr", "C:/Users/jamie/Desktop");
	//setEnvironmentTexture(environmentTexture);

	MaterialStruct ground{};
	ground.type = Material::BRDF_DIFFUSE;
	ground.baseColor = { 0.05f };
	ground.roughness = 1.0f;

	MaterialStruct diffuse{};
	diffuse.type = Material::BRDF_DIFFUSE;
	diffuse.baseColor = { 0.8f };
	diffuse.roughness = 0.8f;

	MaterialStruct micro{};
	micro.type = Material::BRDF_MICROFACET;
	micro.baseColor = { 0.8f };
	//micro.roughness = 0.5f;
	//micro.roughness = 1.0f;
	micro.roughness = 0.0f;

	MaterialStruct lambert{};
	lambert.type = Material::BRDF_LAMBERT;
	lambert.baseColor = { 0.8f };
	lambert.roughness = 0.0f;

	std::vector<std::tuple<std::string, MaterialStruct>> mats{
		{"ground", ground},
		{"diffuse", diffuse },
		{"microfacet", micro },
		{"lambert", lambert }
	};

	SL_LOG("PLEASE SELECT A MATERIAL FOR MESH");
	for (uint32_t i{ 0 }; i < mats.size(); i++)
		fmt::print("{} [{}]\n", std::get<std::string>(mats[i]), i);

	for (uint32_t i{ 0 }; i < meshData.size(); i++)
	{
		fmt::print("{}: ", meshNames[i]);
		std::cin >> meshData[i]->materialId;
		add(meshData[i]);
	}

	//for (uint32_t i{ 0 }; i < meshData.size(); i++)
	//{
	//	meshData[i]->materialId = 0;
	//	add(meshData[i]);
	//}

	std::vector<MaterialStruct> materials{};
	for (auto& e : mats)
		materials.push_back(std::get<MaterialStruct>(e));

	render(cam, materials);

	Image result{ cam.fbSize.x, cam.fbSize.y, (const uint32_t*)owlBufferGetPointer(renderer.frameBuffer, 0) };
	writeImage(result, fmt::format("{}{}.png", prefixPath, "imgae"));
	//writeImage(result, fmt::format("{}{}.png", prefixPath, "rough"));


	release();
}

// 1. load materials
// 2. load obj
// 3. reference mesh to material
// 4. add mesh & material to renderer