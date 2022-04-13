#ifndef BA_BXDF_HPP
#define BA_BXDF_HPP

#include "Types.hpp"
#include "Sampling.hpp"
#include <owl/common/math/vec.h>

struct Material
{
	owl::vec3f baseColor;
	owl::vec3f subsurfaceColor;
	float metallic;
	float specular;
	float roughness;
	float specular_tint;
	float anisotropy;
	float sheen;
	float sheen_tint;
	float clearcoat;
	float clearcoat_gloss;
	float specular_transmission;
	float transmission_roughness;
	float flatness;
	float alpha;
};

inline __device__ bool sameHemisphere(owl::vec3f const& wo, owl::vec3f const& wi, owl::vec3f const& n)
{
	return owl::dot(wo, n) * owl::dot(wi, n) > 0.0f;
}


#endif // BA_BXDF_HPP