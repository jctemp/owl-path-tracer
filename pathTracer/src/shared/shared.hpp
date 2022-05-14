#ifndef SHARED_HPP
#define SHARED_HPP
#pragma once

#include "macros.hpp"
#include "random.hpp"
#include "types.hpp"
#include <owl/owl.h>
#include <cuda_runtime.h>

enum class Material
{
	NONE = 0 << 0,
	LAMBERT = 1 << 0,
	DISNEY = 1 << 0,
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

#endif // !SHARED_HPP
