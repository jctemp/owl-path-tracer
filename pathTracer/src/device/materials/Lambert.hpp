#ifndef LAMBERT_HPP
#define LAMBERT_HPP

#include "../SampleMethods.hpp"

namespace Lambert
{
	DEVICE void f(MaterialStruct& ms, Float3 const& V, Float3 const& L, 
		Float3& bsdf)
	{
		if (!sameHemisphere(V, L)) bsdf = 0.0f;
		bsdf = ms.baseColor * INV_PI * absCosTheta(L);
	}

	DEVICE void sampleF(MaterialStruct& ms, Float3 const& V, Float2 u, Float3& L, 
		Float3& bsdf, Float& pdf)
	{
		sampleCosineHemisphere({ u.x ,u.v }, L);
		pdfCosineHemisphere(V, L, pdf);
		f(ms, V, L, bsdf);
	}

	DEVICE void pdf(Float3 const& V, Float3 const& L, 
		Float& pdf)
	{
		if (!sameHemisphere(V, L)) pdf = 0.0f;
		else pdfCosineHemisphere(V, L, pdf);
	}
}


#endif // !LAMBERT_HPP
