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

	auto const [meshNames, meshData] {loadOBJ(prefixPath + "scenes/mitsuba.obj")};
	Camera cam{
		{ 600 },		  // image size
		{5.0f,3.0f,0.0f}, // look from
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
	//	{ 1024 },		  // image size
	//	{2.0f,1.2f,0.0f}, // look from
	//	{0.0f,0.5f,0.0f}, // look at
	//	{0.0f,1.0f,0.0f}, // look up
	//	50.0f			  // vfov
	//};


	renderer.frameBuffer = owlHostPinnedBufferCreate(
		renderer.context, OWL_INT, cam.fbSize.x * cam.fbSize.y);
	renderer.useEnvironmentMap = true;
	renderer.samplesPerPixel = 1024;
	renderer.maxDepth = 5;

	//ImageRgb environmentTexture{};
	//loadImage(environmentTexture, "env.hdr", "C:/Users/jamie/Desktop");
	//setEnvironmentTexture(environmentTexture);

	/* STANDARD MATERIALS */
	MaterialStruct default1{ Material::DISNEY_DIFFUSE };
	default1.baseColor = { 0.5f };
	MaterialStruct default2{ Material::DISNEY_DIFFUSE };
	default2.baseColor = { 0.15f };

	/* DIFFUSE */
	MaterialStruct diffuse{Material::DISNEY_DIFFUSE};
	diffuse.baseColor = { 1.0f, 0.364f, 0.084f };

	/* FAKE SUBSURFACE */
	MaterialStruct fakeSubsurface{ Material::DISNEY_FAKE_SS };
	fakeSubsurface.subsurface = 1.0f;

	/* RETRO */
	MaterialStruct retro{ Material::DISNEY_RETRO };

	/* SHEEN */
	MaterialStruct sheen{ Material::DISNEY_SHEEN };
	sheen.sheen = 1.0f;

	/* CLEARCOAT */
	MaterialStruct clearcoat{ Material::DISNEY_CLEARCOAT };
	clearcoat.clearcoat = 1.0f;
	//clearcoat.clearcoatGloss = 1.0f;
	//clearcoat.clearcoatGloss = 0.0f;

	/* MICROFACET */
	MaterialStruct microfacet{ Material::DISNEY_MICROFACET };
	microfacet.roughness = 0.1f;
	microfacet.anisotropic = 0.9f;


	std::vector<std::tuple<std::string, MaterialStruct>> mats{
		{"default1", default1},
		{"default2", default2},
		{"diffuse", diffuse},
		{"fakeSubsurface", fakeSubsurface},
		{"retro", retro},
		{"sheen", sheen},
		{"clearcoat", clearcoat},
		{"microfacet", microfacet}
	};

	SL_LOG("===================================");
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
	writeImage(result, fmt::format("{}{}.png", prefixPath, "microfacetAnisotropic"));
	//writeImage(result, fmt::format("{}{}.png", prefixPath, "rough"));


	release();
}

// 1. load materials
// 2. load obj
// 3. reference mesh to material
// 4. add mesh & material to renderer