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
	Float intensity{ 1.f };
	Float exposure{ 0.f };
	Float falloff{ 2.0f };
	bool use_surface_area = false;
};

struct launch_params_data
{
	Int max_path_depth;
	Int max_samples;
	Buffer material_buffer;
	Buffer light_buffer;
	OptixTraversableHandle world;
	cudaTextureObject_t environment_map;
	bool use_environment_map;
};

struct triangle_geom_data
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
	camera_data camera;
};

struct MissProgData
{
};

#endif // !SHARED_HPP
