#ifndef GLOBALS_HPP
#define GLOBALS_HPP
#pragma once

#include <owl/owl.h>
#include <owl/owl_device_buffer.h>
#include <owl/common/math/LinearSpace.h>
#include <owl/common/math/vec.h>
#include <cuda_runtime.h>

using Float = float;
using Float2 = owl::vec2f;
using Float3 = owl::vec3f;
using Float4 = owl::vec4f;

using Int = int32_t;
using Int2 = owl::vec2i;
using Int3 = owl::vec3i;

using Uint = uint32_t;
using Uint2 = owl::vec2ui;
using Uint3 = owl::vec3ui;

using Buffer = owl::device::Buffer;

enum class Material
{
	NONE = 0 << 0,
	BRDF_LAMBERT = 1 << 0,
	DISNEY_DIFFUSE = 1 << 1,
	DISNEY_FAKE_SS = 1 << 2,
	DISNEY_RETRO = 1 << 3,
	DISNEY_SHEEN = 1 << 4,
	DISNEY_CLEARCOAT = 1 << 5,
	DISNEY_MICROFACET = 1 << 6,
	DISNEY_BRDF = 1 << 7,
	EMISSION = 1 << 8
};

struct MaterialStruct
{
	Material type{ Material::NONE };
	Float3   baseColor{ 0.8f, 0.8f, 0.8f };
	Float    subsurface{ 0.0f };
	Float3   subsurfaceRadius{ 1.0f, 0.2f, 0.1f };
	Float3   subsurfaceColor{ 0.8f, 0.8f, 0.8f };
	Float    metallic{ 0.0f };
	Float    specular{ 0.5f };
	Float    specularTint{ 1.0f };
	Float    roughness{ 0.5f };
	Float    sheen{ 0.0f };
	Float    sheenTint{ 1.0f };
	Float    clearcoat{ 0.0f };
	Float    clearcoatGloss{ 0.03f };
	Float    ior{ 1.45f };
	Float    transmission{ 0.0f };
	Float    transmissionRoughness{ 0.0f };
};

enum class Light
{
	NONE = 0 << 0,
	MESH = 1 << 0
};

struct LightStruct
{
	Light type{ Light::MESH };
	Float3 color{ 1.f };
	Float intensity{ 1.f };
	Float exposure{ 0.f };
	Float falloff{ 2.0f };
	bool use_surface_area = false;
};

struct LaunchParams
{
	Int maxDepth;
	Int samplesPerPixel;
	Buffer materials;
	Buffer lights;
	OptixTraversableHandle world;
	cudaTextureObject_t environmentMap;
	bool useEnvironmentMap;
};

struct TrianglesGeomData
{
	Int matId{ -1 };
	Int lightId{ -1 };
	Int3* index;
	Float3* vertex;
	Float3* normal;
};

struct RayGenData
{
	Uint* fbPtr;
	Int2 fbSize;

	struct
	{
		Float3 origin;
		Float3 llc;
		Float3 horizontal;
		Float3 vertical;
	} camera;
};

struct MissProgData
{
};

#endif // GLOBALS_HPP