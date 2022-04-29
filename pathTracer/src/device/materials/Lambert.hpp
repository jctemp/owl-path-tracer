#ifndef LAMBERT_HPP
#define LAMBERT_HPP
#pragma once

#include "Materials.hpp"
#include "../Sampling.hpp"

template<>
DEVICE void f<Material::BRDF_LAMBERT>(MaterialStruct& ms, Float3 const& V, Float3 const& L,
	Float3& bsdf)
{
	if (!sameHemisphere(V, L)) bsdf = 0.0f;
	bsdf = ms.baseColor * INV_PI;
}

template<>
DEVICE void sampleF<Material::BRDF_LAMBERT>(MaterialStruct& ms, Float3 const& V, Float2 u, Float3& L,
	Float3& bsdf, Float& pdf)
{
	sampleCosineHemisphere({ u.x ,u.v }, L);
	pdfCosineHemisphere(V, L, pdf);
	f<Material::BRDF_LAMBERT>(ms, V, L, bsdf);
}

template<>
DEVICE void pdf<Material::BRDF_LAMBERT>(Float3 const& V, Float3 const& L,
	Float& pdf)
{
	if (!sameHemisphere(V, L)) pdf = 0.0f;
	else pdfCosineHemisphere(V, L, pdf);
}


#endif // !LAMBERT_HPP
