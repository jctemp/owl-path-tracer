#ifndef DISNEY_BRDF_HPP
#define DISNEY_BRDF_HPP

#include "Materials.hpp"
#include "../Sampling.hpp"

// REFECRENCES:
// https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// 


DEVICE_INL Float schlickFresnel(Float cosTheta)
{
	return pow(saturate(1.f - cosTheta), 5.f);
}


/*
 * DISNEY PRINCIPLED DIFFUSE 
 *
 * f_d = \frac{\text{baseColor}}{\pi} (1 + (F_{D90} - 1)(1 - \cos\theta_l)^5 (1 + (F_{D90} - 1)(1 - \cos\theta_v)^5 )
 * with
 * F_{D90} = 0.5 + 2 \text{roughness}\cos^2\theta_d
 *
 */

template<>
DEVICE void f<Material::BRDF_DIFFUSE>(MaterialStruct& ms, Float3 const& V, Float3 const& L,
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

	// one could change the baseColor to linear space with pow(color, 2.2f)
	brdf = Float3{ ms.baseColor * INV_PI * fd };
}

template<>
DEVICE void sampleF<Material::BRDF_DIFFUSE>(MaterialStruct& ms, Float3 const& V, Float2 u, Float3& L,
	Float3& brdf, Float& pdf)
{
	sampleCosineHemisphere({ u.x ,u.v }, L);
	pdfCosineHemisphere(V, L, pdf);
	Float3 bsdfDiffuse{};
	f<Material::BRDF_DIFFUSE>(ms, V, L, brdf);
}

template<>
DEVICE void pdf<Material::BRDF_DIFFUSE>(Float3 const& V, Float3 const& L,
	Float& pdf)
{
	if (!sameHemisphere(V, L)) pdf = 0.0f;
	else pdfCosineHemisphere(V, L, pdf);
}

/*
* DISNEY PRINCIPLED GLOSSY 
*
*
*/

template<>
DEVICE void f<Material::BRDF_MICROFACET>(MaterialStruct& ms, Float3 const& V, Float3 const& L,
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

	// one could change the baseColor to linear space with pow(color, 2.2f)
	brdf = Float3{ ms.baseColor * INV_PI * fd };
}

template<>
DEVICE void sampleF<Material::BRDF_MICROFACET>(MaterialStruct& ms, Float3 const& V, Float2 u, Float3& L,
	Float3& brdf, Float& pdf)
{
	sampleCosineHemisphere({ u.x ,u.v }, L);
	pdfCosineHemisphere(V, L, pdf);
	Float3 bsdfDiffuse{};
	f<Material::BRDF_MICROFACET>(ms, V, L, brdf);
}

template<>
DEVICE void pdf<Material::BRDF_MICROFACET>(Float3 const& V, Float3 const& L,
	Float& pdf)
{
	if (!sameHemisphere(V, L)) pdf = 0.0f;
	else pdfCosineHemisphere(V, L, pdf);
}




#endif // !DISNEY_BRDF_HPP
