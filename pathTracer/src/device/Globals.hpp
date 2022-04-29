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
	BRDF_LAMBERT = 1 << 0,
	BRDF_DIFFUSE = 1 << 1,
	BRDF_MICROFACET = 1 << 2
};

enum class Microfacet
{
	GGX = 1 << 0
};

struct MaterialStruct
{
	Material type = Material::BRDF_LAMBERT;
	Float3 baseColor = Float3{ 0.8f, 0.8f, 0.8f };
	Float subsurface = 0.0f;
	Float3 subsurfaceRadius = Float3{ 1.0f, 0.2f, 0.1f };
	Float3 subsurfaceColor = Float3{ 0.8f, 0.8f, 0.8f };
	Float metallic = 0.0f;
	Float specular = 0.5f;
	Float specularTint = 0.0f;
	Float roughness = 0.5f;
	Float anisotropic = 0.0f;
	Float anisotropicRotation = 0.0f;
	Float sheen = 0.0f;
	Float sheenTint = 0.5f;
	Float clearcoat = 0.0f;
	Float clearcoatRoughness = 0.03f;
	Float ior = 1.45f;
	Float transmission = 0.0f;
	Float transmissionRoughness = 0.0f;
};

struct LaunchParams
{
	Int maxDepth;
	Int samplesPerPixel;
	Buffer materials;
	OptixTraversableHandle world;
	cudaTextureObject_t environmentMap;
	bool useEnvironmentMap;
};

struct TrianglesGeomData
{
	Uint matId;
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