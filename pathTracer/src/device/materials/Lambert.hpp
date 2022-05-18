#ifndef LAMBERT_HPP
#define LAMBERT_HPP
#pragma once

#include "types.hpp"
#include "../Sampling.hpp"

__device__ vec3 fLambert(material_data const& mat, vec3 const& V, vec3 const& L)
{
	if (!sameHemisphere(V, L)) return vec3{ 0.0f };
	return mat.baseColor * inv_pi;
}

__device__ float pdfLambert(material_data const& mat, vec3 const& V, vec3 const& L)
{
	if (!sameHemisphere(V, L)) return 0.0f;
	float pdf{ 0.0f };
	pdfCosineHemisphere(V, L, pdf);
	return pdf;
}

__device__ void sampleLambert(material_data const& mat, vec3 const& V, vec3& L,
	Random& rand, vec3& bsdf, float& pdf)
{
	sampleCosineHemisphere({ rand.random() , rand.random()}, L);
	pdf = pdfLambert(mat, V, L);
	bsdf = fLambert(mat, V ,L);
}

#endif // !LAMBERT_HPP
