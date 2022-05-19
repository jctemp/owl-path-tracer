#ifndef DEVICE_HPP
#define DEVICE_HPP
#pragma once

#include <cuda_runtime.h>
#include "camera.hpp"
#include "random.hpp"

struct material_data
{
    enum class type
    {
        NONE = 0 << 0,
        LAMBERT = 1 << 0,
        DISNEY = 1 << 0,
    };

    type type{ type::NONE };
    vec3  baseColor{ 0.8f, 0.8f, 0.8f };
    float subsurface{ 0.0f };
    vec3  subsurfaceRadius{ 1.0f, 0.2f, 0.1f };
    vec3  subsurfaceColor{ 0.8f, 0.8f, 0.8f };
    float metallic{ 0.0f };
    float specular{ 0.5f };
    float specularTint{ 1.0f };
    float roughness{ 0.5f };
    float sheen{ 0.0f };
    float sheenTint{ 1.0f };
    float clearcoat{ 0.0f };
    float clearcoatGloss{ 0.03f };
    float ior{ 1.45f };
};

struct light_data
{
    enum class type
    {
        NONE = 0 << 0,
        MESH = 1 << 0
    };

    type type{ type::MESH };
    vec3 color{ 1.f };
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
    ivec3* index;
    vec3* vertex;
    vec3* normal;
};

struct ray_gen_data
{
    uint32_t* fbPtr;
    ivec2 fbSize;
    camera_data camera;
};

struct miss_data
{
};

enum class ScatterEvent
{
	BOUNCED = 1 << 0,
	CANCELLED = 1 << 1,
	MISS = 1 << 2,
	NONE = 1 << 3
};

struct InterfaceStruct
{
	/* triangle points */
	vec3 TRI[3];

	/* hit position */
	vec3 P;

	/* shading normal */
	vec3 N;

	/* geometric normal */
	vec3 Ng;

	/* view direction (wo or V) */
	vec3 V;

	/* barycentrics */
	vec2 UV;

	/* thit */
	float t;

	/* primitive id => 0 if not exists */
	int32_t prim;

	/* material id for LP reference */
	int32_t matId;

	/* light id for LP reference */
	int32_t lightId;
};

struct per_ray_data
{
	Random& random;
	ScatterEvent scatterEvent;
	InterfaceStruct* is;
	material_data* ms;
};

#endif // !DEVICE_HPP
