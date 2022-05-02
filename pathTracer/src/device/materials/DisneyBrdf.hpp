#ifndef DISNEY_BRDF_HPP
#define DISNEY_BRDF_HPP
#pragma once

#include "Materials.hpp"
#include "../Sampling.hpp"
#include "DisneyBrdfUtils.hpp"

// REFECRENCES:
// https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           DISNEY'S PRINCIPLED F


/* DIFFUSE */
DEVICE Float3 disneyDiffuseF(MaterialStruct const& mat, Float3 const& N, Float3 const& V,
	Float3 const& L)
{
	Float NdotV{ absCosTheta(V) };
	Float NdotL{ absCosTheta(L) };

	Float FL{ schlickFresnel(NdotL) };
	Float FV{ schlickFresnel(NdotV) };

	// Burley 2015, eq (4).
	return mat.baseColor * INV_PI * (1.0f - FL / 2.0f) * (1.0f - FV / 2.0f);
}

/* FAKE SSS */
DEVICE Float3 disneyFakeSubsurfaceF(MaterialStruct const& mat, Float3 const& N, Float3 const& V,
	Float3 const& L)
	// Hanrahan - Krueger BRDF approximation of the BSSRDF
{
	Float3 H{ L + V };
	if (H.x == 0 && H.y == 0 && H.z == 0) return Float3{ 0.0f };

	H = normalize(H);
	Float cosThetaD{ dot(L, H) };
	Float NdotV{ absCosTheta(V) };
	Float NdotL{ absCosTheta(L) };

	// Fss90 used to "flatten" retroreflection based on roughness
	Float Fss90{ cosThetaD * cosThetaD * mat.roughness };
	Float FL{ schlickFresnel(NdotL) };
	Float FV{ schlickFresnel(NdotV) };
	Float Fss{ mix(1.0f, Fss90, FL) * mix(1.0f, Fss90, FV) };

	// 1.25 scale is used to (roughly) preserve albedo
	Float Fs{ 1.25f * (Fss * (1.0f / (NdotV + NdotL) - 0.5f) + 0.5f) };

	return mat.subsurfaceColor * INV_PI * Fs;
}

/* RETRO-REFLECTION */
DEVICE Float3 disneyRetroF(MaterialStruct const& mat, Float3 const& N, Float3 const& V,
	Float3 const& L)
{
	Float3 H{ L + V };
	if (H.x == 0 && H.y == 0 && H.z == 0) return Float3{ 0.0f };

	H = normalize(H);
	Float cosThetaD{ dot(L, H) };
	Float NdotV{ absCosTheta(V) };
	Float NdotL{ absCosTheta(L) };

	Float FL{ schlickFresnel(NdotL) };
	Float FV{ schlickFresnel(NdotV) };
	Float Rr{ 2 * mat.roughness * cosThetaD * cosThetaD };

	// Burley 2015, eq (4).
	return mat.baseColor * INV_PI * Rr * (FL + FV + FL * FV * (Rr - 1));
}





/* SHEEN */
DEVICE Float3 evalDisneySheen(MaterialStruct const& mat, Float3 const& V, Float3 const& L,
	Float3 const& H)
	// - adds back energy for rough surfaces or cloth due to geometric term approx.
	// - smithG is not energy conserving for multi-scattering due to over-estimation
	// - additive
{
	if (mat.sheen <= 0.0f) return Float3{ 0.0f };

	Float HdotL{ dot(H, L) };
	Float3 tint{ calculateTint(mat.baseColor) };
	return mat.sheen * mix(Float3{ 1.0f }, tint, mat.sheenTint) * schlickFresnel(HdotL);
}


/* CLEARCOAT */
DEVICE Float evalDisneyClearcoat(Float clearcoat, Float alpha, Float3 const& V, Float3 const& L,
	Float3 const& H, Float& pdf)
	// - Burley found that Trowbridge-Reitz distrubtuion is not perfect fitting for most materials
	// - propose another specular component with modified NDF to add to BSDF model
	// - ad-hoc fit with no clear geometric meaning 
	// - additive
{
	if (clearcoat <= 0.0f) {
		return 0.0f;
	}

	Float NdotV{ absCosTheta(V) };
	Float NdotL{ absCosTheta(L) };
	Float NdotH{ absCosTheta(H) };
	Float LdotH{ dot(L, H) };

	Float d{ gtr1(NdotH, mix(0.1f, 0.001f, alpha)) };
	Float f{ schlickFresnel(0.04f, LdotH) };
	Float gl{ smithG(NdotL, 0.25f) };
	Float gv{ smithG(NdotV, 0.25f) };

	pdf = d / (4.0f * NdotL);

	return 0.25f * clearcoat * d * f * gl * gv;
}


/* SPECULAR */
DEVICE Float3 evalDisneyMicrofacetIsotropic(MaterialStruct const& mat, Float3 const& N, Float3 const& V,
	Float3 const& L, Float3 const& H, Float pdf)
{
	Float3 tint{ calculateTint(mat.baseColor) };
	Float3 spec{ mix(mat.specular * 0.08f * mix(Float3{1.0f}, tint, mat.specularTint), 
		mat.baseColor, mat.metallic) };

	Float NdotV{ absCosTheta(V) };
	Float NdotL{ absCosTheta(L) };
	Float NdotH{ absCosTheta(H) };

	Float alpha{ roughnessToAlpha(mat.roughness) };
	Float Ds{ gtr2(NdotH, alpha) };
	Float Fs{ disneyFresnel(mat, V, L, H) };
	Float Vs{ smithVCorrelated(NdotV, NdotL, alpha)};

	pdf = pdfGtr2(V, L, H, alpha);

	return Float3{ Ds * Fs * Vs };
}


/* TRANSMISSIVE */
DEVICE Float3 evalDisneyMicrofacetTransmission(MaterialStruct const& mat, Float3 const& N,
	Float3 const& V, Float3 const& L, Float3 const& H, Float& pdf)
{
	printf("NOT IMPLEMENTED");
}



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           DISNEY'S PRINCIPLED BSDF
//
//  Specular BSDF ───────┐ Metallic BRDF ┐
//                       │               │
//  Dielectric BRDF ─── MIX ─────────── MIX ─── DISNEY BSDF  
//  + Subsurface
//


DEVICE void sampleDisneyDiffuse(MaterialStruct const& mat, Float3 const& N, Float3 const& V,
	Float3& L, Random& rand, Float3& bsdf, Float& pdf)
{
	sampleCosineHemisphere({ rand.random() ,rand.random() }, L);
	pdfCosineHemisphere(V, L, pdf);
	bsdf = disneyDiffuseF(mat, N, V, L) * absCosTheta(L);
}

DEVICE void sampleDisneyFakeSubsurface(MaterialStruct const& mat, Float3 const& N, Float3 const& V,
	Float3& L, Random& rand, Float3& bsdf, Float& pdf)
{
	sampleCosineHemisphere({ rand.random() ,rand.random() }, L);
	pdfCosineHemisphere(V, L, pdf);
	bsdf = disneyFakeSubsurfaceF(mat, N, V, L) * absCosTheta(L);
}

DEVICE void sampleDisneyRetro(MaterialStruct const& mat, Float3 const& N, Float3 const& V,
	Float3& L, Random& rand, Float3& bsdf, Float& pdf)
{
	sampleCosineHemisphere({ rand.random() ,rand.random() }, L);
	pdfCosineHemisphere(V, L, pdf);
	bsdf = disneyRetroF(mat, N, V, L) * absCosTheta(L);
}








DEVICE void sampleDisneyMicrofacet(MaterialStruct const& mat, Float3 const& N, Float3 const& V,
	Float3& L, Random& rand, Float3& bsdf, Float& pdf)
{
}




DEVICE void getLobeWeights(MaterialStruct const& mat, Float& pSpecular, Float& pDiffuse,
	Float& pClearcoat, Float& pSpecTrans)
{

}


DEVICE void evalDisneyBSDF(MaterialStruct const& mat, Float3 const& N, Float3 const& V,
	Float3 const& L, Float3& bsdf, Float& pdf)

{
	Float3 H{ normalize(V + L) };
	Float NdotV{ cosTheta(V) };
	Float NdotL{ cosTheta(L) };

	pdf = 0.0f;
	bsdf = { 0.0f };
	//Float diffuseWeight, specReflectWeight, specRefractWeight, clearcloatWeight;
	Float2 alpha{ roughnessToAlpha(mat.roughness, mat.anisotropic) };
	bool upperHemisphere{ NdotV > 0.0f && NdotL > 0.0f };
}





/*
* MICROFACET MODELS
*
* f(l, v) = \text{diffuse} + \frac{D(\theta_h)F(\theta_d)G(\theta_l, \theta_v)}{4\cos\theta_l\cos\theta_v}
* - D is responsible for specular peak				=> GGX distribution
* - F is reflection coefficient (fresnel)			=> fresnel-schlick approximation
* - G is geometric attenuation or shadowing factor	=> GGX shadowing or visibility term (self-shadowing)
* => using GGX because of short peaks and long tails for specular
*
* - \theta_l and \theta_v angles of incident of in coming and out going vector (l, v)
* - \theta_h angle between half-vector and normal
* - \theta_d differens between l and h or v and h
*
* Microfacet model is energy conserving allows for one parameter roughness! Defines behaviour for
* specular reflections or refractions of an object.
* The NDF is dominant component in the model which drives the roughness appearance most.
*
* 1. Sample random microfacet normal
* 2. reflect exitence radiance along normal to get incident direction (reflect V to get L)
*
* Different Types:
*	- GGX
*	- Beckmann
*	- Blinn
*/


#endif // !DISNEY_BRDF_HPP
