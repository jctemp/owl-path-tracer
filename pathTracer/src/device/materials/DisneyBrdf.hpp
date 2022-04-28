#ifndef DISNEY_BRDF_HPP
#define DISNEY_BRDF_HPP

#include "../SampleMethods.hpp"

// REFECRENCES:
// https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// 

/*
* Microfacet Model
* 
* f(l, v) = \text{diffuse} + \frac{D(\theta_h)F(\theta_d)G(\theta_l, \theta_v)}{4\cos\theta_l\cos\theta_v}
* - D is responsible for specular peak				=> GGX distribution
* - F is reflection coefficient (fresnel)			=> fresnel-schlick approximation
* - G is geometric attenuation or shadowing factor	=> GGX shadowing or visibility term
* => using GGX because of short peaks and long tails for specular
* 
* - \theta_l and \theta_v angles of incident of in coming and out going vector (l, v)
* - \theta_h angle between half-vector and normal
* - \theta_d differens between l and h or v and h
* 
*/



/*
* Diffuse Model	
* 
* f_d = \frac{\text{baseColor}}{\pi} (1 + (F_{D90} - 1)(1 - \cos\theta_l)^5 (1 + (F_{D90} - 1)(1 - \cos\theta_v)^5 )
* with
* F_{D90} = 0.5 + 2 \text{roughness}\cos^2\theta_d
* 
*/

DEVICE_INL Float schlickFresnel(Float cosTheta)
{
	return pow(saturate(1.f - cosTheta), 5.f);
}

DEVICE_INL Float luminance(Float3 color)
{
	return 0.212671f * color.x + 0.715160f * color.y + 0.072169f * color.z;
}

DEVICE_INL Float3 pow(Float3 const& base, Float3 const& exponent)
{
	return {
		powf(base.x, exponent.x),
		powf(base.y, exponent.y),
		powf(base.z, exponent.z)
	};
}


DEVICE_INL Float3 linearToGamma(Float3 linearColor) {
	return pow(linearColor, Float3{ 0.4545f });
}


DEVICE_INL Float3 gammaToLinear(Float3 gammaColor) {
	return pow(gammaColor, Float3{ 2.2f });
}


namespace Diffuse
{
	DEVICE void f(MaterialStruct& ms, Float3 const& V, Float3 const& L,
		Float3& brdf)
	{
		Float3 H = normalize(V + L);
		Float NdotV = absCosTheta(V);
		Float NdotL = absCosTheta(L);
		Float LdotH = dot(L, H);

		Float fd90 = 0.5f + 1.0f * ms.roughness * LdotH * LdotH;
		Float fl = schlickFresnel(NdotL);
		Float fv = schlickFresnel(NdotV);


		Float fd = mix(1.f, fd90, fl) * mix(1.f, fd90, fv);

		brdf = Float3{ INV_PI * fd * absCosTheta(L) };
	}

	DEVICE void sampleF(MaterialStruct& ms, Float3 const& V, Float2 u, Float3& L,
		Float3& bsdf, Float& pdf)
	{
		sampleCosineHemisphere({ u.x ,u.v }, L);
		pdfCosineHemisphere(V, L, pdf);
		Float3 bsdfDiffuse{};
		f(ms, V, L, bsdfDiffuse);

		//auto cdLin = gammaToLinear(ms.baseColor);

		// why do I have to multiply that with dot(N,L) ???
		bsdf = bsdfDiffuse * ms.baseColor;

	}

	DEVICE void pdf(Float3 const& V, Float3 const& L,
		Float& pdf)
	{
		if (!sameHemisphere(V, L)) pdf = 0.0f;
		else pdfCosineHemisphere(V, L, pdf);
	}
}


#endif // !DISNEY_BRDF_HPP
