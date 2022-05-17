#ifndef LAMBERT_HPP
#define LAMBERT_HPP
#pragma once

#include "../Sampling.hpp"

PT_DEVICE Float3 fLambert(material_data const& mat, Float3 const& V, Float3 const& L)
{
	if (!sameHemisphere(V, L)) return Float3{ 0.0f };
	return mat.baseColor * INV_PI;
}

PT_DEVICE float pdfLambert(material_data const& mat, Float3 const& V, Float3 const& L)
{
	if (!sameHemisphere(V, L)) return 0.0f;
	float pdf{ 0.0f };
	pdfCosineHemisphere(V, L, pdf);
	return pdf;
}

PT_DEVICE void sampleLambert(material_data const& mat, Float3 const& V, Float3& L,
	Random& rand, Float3& bsdf, float& pdf)
{
	sampleCosineHemisphere({ rand.random() , rand.random()}, L);
	pdf = pdfLambert(mat, V, L);
	bsdf = fLambert(mat, V ,L);
}

#endif // !LAMBERT_HPP
