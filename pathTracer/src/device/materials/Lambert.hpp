#ifndef LAMBERT_HPP
#define LAMBERT_HPP
#pragma once

#include "../Sampling.hpp"

DEVICE Float3 fLambert(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	if (!sameHemisphere(V, L)) return Float3{ 0.0f };
	return mat.baseColor * INV_PI;
}

DEVICE Float pdfLambert(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	if (!sameHemisphere(V, L)) return 0.0f;
	Float pdf{ 0.0f };
	pdfCosineHemisphere(V, L, pdf);
	return pdf;
}

DEVICE void sampleLambert(MaterialStruct const& mat, Float3 const& V, Float3& L,
	Random& rand, Float3& bsdf, Float& pdf)
{
	sampleCosineHemisphere({ rand.random() , rand.random()}, L);
	pdf = pdfLambert(mat, V, L);
	bsdf = fLambert(mat, V ,L);
}

#endif // !LAMBERT_HPP
