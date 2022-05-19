#ifndef LAMBERT_HPP
#define LAMBERT_HPP
#pragma once

#include "types.hpp"
#include "../../sample_methods.hpp"

__device__ vec3 fLambert(material_data const& mat, vec3 const& V, vec3 const& L)
{
	if (!same_hemisphere(V, L)) return vec3{0.0f };
	return mat.base_color * inv_pi;
}

__device__ float pdfLambert(material_data const& mat, vec3 const& V, vec3 const& L)
{
	if (!same_hemisphere(V, L)) return 0.0f;
	float pdf{ 0.0f };
    pdf_cosine_hemisphere(V, L);
	return pdf;
}

__device__ void sampleLambert(material_data const& mat, vec3 const& V, vec3& L,
	Random& rand, vec3& bsdf, float& pdf)
{
    sample_cosine_hemisphere({rand.random(), rand.random()});
	pdf = pdfLambert(mat, V, L);
	bsdf = fLambert(mat, V ,L);
}

#endif // !LAMBERT_HPP
