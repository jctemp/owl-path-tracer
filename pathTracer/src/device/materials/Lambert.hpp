#ifndef LAMBERT_HPP
#define LAMBERT_HPP

#include "../SampleMethods.hpp"

namespace Lambert
{
	DEVICE Float3 f(ShadingData& sd, Float3 const& V, Float3 const& L)
	{
		if (!sameHemisphere(V, L)) return 0.0f;
		return sd.md->baseColor * INV_PI * absCosTheta(L);
	}

	DEVICE Float3 sampleF(ShadingData& sd, Float3 const& V, Float3& L, Float& pdf)
	{
		sampleCosineHemisphere({ sd.random() ,sd.random() }, L);
		pdfCosineHemisphere(V, L, pdf);
		return f(sd, V, L);
	}

	DEVICE void pdf(Float3 const& V, Float3 const& L, Float& pdf)
	{
		if (!sameHemisphere(V, L)) pdf = 0.0f;
		else pdfCosineHemisphere(V, L, pdf);
	}
}


#endif // !LAMBERT_HPP
