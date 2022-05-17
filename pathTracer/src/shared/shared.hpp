#ifndef SHARED_HPP
#define SHARED_HPP
#pragma once

#include "macros.hpp"
#include "random.hpp"
#include "types.hpp"
#include <owl/owl.h>
#include <cuda_runtime.h>

#include <device/camera.hpp>

struct material_data
{
	enum class type
	{
		NONE = 0 << 0,
		LAMBERT = 1 << 0,
		DISNEY = 1 << 0,
	};

	type type{ type::NONE };
	Float3   baseColor{ 0.8f, 0.8f, 0.8f };
	float    subsurface{ 0.0f };
	Float3   subsurfaceRadius{ 1.0f, 0.2f, 0.1f };
	Float3   subsurfaceColor{ 0.8f, 0.8f, 0.8f };
	float    metallic{ 0.0f };
	float    specular{ 0.5f };
	float    specularTint{ 1.0f };
	float    roughness{ 0.5f };
	float    sheen{ 0.0f };
	float    sheenTint{ 1.0f };
	float    clearcoat{ 0.0f };
	float    clearcoatGloss{ 0.03f };
	float    ior{ 1.45f };
};

struct light_data
{
	enum class type
	{
		NONE = 0 << 0,
		MESH = 1 << 0
	};

	type type{ type::MESH };
	Float3 color{ 1.f };
	float intensity{ 1.f };
	float exposure{ 0.f };
	float falloff{ 2.0f };
	bool use_surface_area = false;
};

struct launch_params_data
{
	int32_t max_path_depth;
	int32_t max_samples;
	Buffer material_buffer;
	Buffer light_buffer;
	OptixTraversableHandle world;
	cudaTextureObject_t environment_map;
	bool use_environment_map;
};

struct triangle_geom_data
{
	int32_t matId{ -1 };
	int32_t lightId{ -1 };
	Int3* index;
	Float3* vertex;
	Float3* normal;
};

struct RayGenData
{
	uint32_t* fbPtr;
	Int2 fbSize;
	camera_data camera;
};

struct MissProgData
{
};

#endif // !SHARED_HPP
