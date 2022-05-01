#ifndef LAMBERT_HPP
#define LAMBERT_HPP
#pragma once

#include "Materials.hpp"
#include "../Sampling.hpp"

template<>
DEVICE void f<Material::BRDF_LAMBERT>(MaterialStruct const& mat, Float3 const& Ng, Float3 const& N,
	Float3 const& T, Float3 const& B, Float3 const& V, Float3 const& L, Float3 const& H,
	Float3& bsdf)
{
	if (!sameHemisphere(V, L)) bsdf = 0.0f;
	bsdf = mat.baseColor * INV_PI;
}

template<>
DEVICE void sampleF<Material::BRDF_LAMBERT>(MaterialStruct const& mat, Random& random, Float3 const& Ng, Float3 const& N,
	Float3 const& T, Float3 const& B, Float3 const& V, Float3& L, Float& pdf, Float3& bsdf)
{
	sampleCosineHemisphere({ random() , random()}, L);
	pdfCosineHemisphere(V, L, pdf);
	f<Material::BRDF_LAMBERT>(mat, Ng, N, T, B, V ,L, {}, bsdf);
}

template<>
DEVICE void pdf<Material::BRDF_LAMBERT>(MaterialStruct const& mat, Float3 const& V, Float3 const& L,
	Float& pdf)
{
	if (!sameHemisphere(V, L)) pdf = 0.0f;
	else pdfCosineHemisphere(V, L, pdf);
}


#endif // !LAMBERT_HPP
