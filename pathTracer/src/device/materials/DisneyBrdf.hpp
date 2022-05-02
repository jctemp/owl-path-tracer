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
//           DISNEY'S COMPONENTS

#pragma region DIFFUSE

DEVICE Float3 fDisneyDiffuse(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	Float NdotV{ absCosTheta(V) };
	Float NdotL{ absCosTheta(L) };

	Float FL{ schlickWeight(NdotL) };
	Float FV{ schlickWeight(NdotV) };

	// Burley 2015, eq (4).
	return mat.baseColor * INV_PI * (1.0f - FL / 2.0f) * (1.0f - FV / 2.0f);
}


DEVICE_INL Float pdfDisneyDiffuse(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	Float pdf{ 0.0f };
	pdfCosineHemisphere(V, L, pdf);
	return pdf;
}


DEVICE void sampleDisneyDiffuse(MaterialStruct const& mat, Float3 const& V, Float3& L,
	Random& rand, Float3& bsdf, Float& pdf)
{
	sampleCosineHemisphere({ rand.random() ,rand.random() }, L);
	pdf = pdfDisneyDiffuse(mat, V, L);
	bsdf = fDisneyDiffuse(mat, V, L) * absCosTheta(L);
}

#pragma endregion

// ──────────────────────────────────────────────────────────────────────────────────────

#pragma region "FAKE SUBSURFACE"

DEVICE Float3 fDisneyFakeSubsurface(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
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
	Float FL{ schlickWeight(NdotL) };
	Float FV{ schlickWeight(NdotV) };
	Float Fss{ mix(1.0f, Fss90, FL) * mix(1.0f, Fss90, FV) };

	// 1.25 scale is used to (roughly) preserve albedo
	Float Fs{ 1.25f * (Fss * (1.0f / (NdotV + NdotL) - 0.5f) + 0.5f) };

	return mat.subsurfaceColor * INV_PI * Fs;
}


DEVICE_INL Float pdfDisneyFakeSubsurface(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	Float pdf{ 0.0f };
	pdfCosineHemisphere(V, L, pdf);
	return pdf;
}


DEVICE void sampleDisneyFakeSubsurface(MaterialStruct const& mat, Float3 const& V, Float3& L,
	Random& rand, Float3& bsdf, Float& pdf)
{
	sampleCosineHemisphere({ rand.random() ,rand.random() }, L);
	pdf = pdfDisneyFakeSubsurface(mat, V, L);
	bsdf = fDisneyFakeSubsurface(mat, V, L) * absCosTheta(L);
}

#pragma endregion

// ──────────────────────────────────────────────────────────────────────────────────────

#pragma region RETRO

DEVICE Float3 fDisneyRetro(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	Float3 H{ L + V };
	if (H.x == 0 && H.y == 0 && H.z == 0) return Float3{ 0.0f };

	H = normalize(H);
	Float cosThetaD{ dot(L, H) };
	Float NdotV{ absCosTheta(V) };
	Float NdotL{ absCosTheta(L) };

	Float FL{ schlickWeight(NdotL) };
	Float FV{ schlickWeight(NdotV) };
	Float Rr{ 2 * mat.roughness * cosThetaD * cosThetaD };

	// Burley 2015, eq (4).
	return mat.baseColor * INV_PI * Rr * (FL + FV + FL * FV * (Rr - 1));
}


DEVICE_INL Float pdfDisneyRetro(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	Float pdf{ 0.0f };
	pdfCosineHemisphere(V, L, pdf);
	return pdf;
}


DEVICE void sampleDisneyRetro(MaterialStruct const& mat, Float3 const& V, Float3& L,
	Random& rand, Float3& bsdf, Float& pdf)
{
	sampleCosineHemisphere({ rand.random() ,rand.random() }, L);
	pdf = pdfDisneyRetro(mat, V, L);
	bsdf = fDisneyRetro(mat, V, L) * absCosTheta(L);
}

#pragma endregion

// ──────────────────────────────────────────────────────────────────────────────────────

#pragma region SHEEN

DEVICE Float3 fDisneySheen(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	Float3 H{ L + V };
	if (H.x == 0 && H.y == 0 && H.z == 0) return Float3{ 0.0f };
	if (mat.sheen <= 0.0f) return Float3{ 0.0f };

	H = normalize(H);
	Float cosThetaD{ dot(L, H) };

	Float3 tint{ calculateTint(mat.baseColor) };
	return mat.sheen * mix(Float3{ 1.0f }, tint, mat.sheenTint) * schlickWeight(cosThetaD);
}


DEVICE_INL Float pdfDisneySheen(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	Float pdf{ 0.0f };
	pdfCosineHemisphere(V, L, pdf);
	return pdf;
}


DEVICE void sampleDisneySheen(MaterialStruct const& mat, Float3 const& V, Float3& L,
	Random& rand, Float3& bsdf, Float& pdf)
{
	sampleCosineHemisphere({ rand.random() ,rand.random() }, L);
	pdf = pdfDisneySheen(mat, V, L);
	bsdf = fDisneySheen(mat, V, L) * absCosTheta(L);
}

#pragma endregion

// ──────────────────────────────────────────────────────────────────────────────────────

#pragma region CLEARCOAT

DEVICE Float3 fDisneyClearcoat(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	Float3 H{ L + V };
	if (H.x == 0 && H.y == 0 && H.z == 0) return Float3{ 0.0f };
	H = normalize(H);

	// Clearcoat has ior = 1.5 hardcoded -> F0 = 0.04. It then uses the
	// GTR1 distribution, which has even fatter tails than Trowbridge-Reitz
	// (which is GTR2). The geometric term always based on alpha = 0.25
	// - pbrt book v3
	Float alphaG{ (1 - mat.clearcoatGloss) * 0.1f + mat.clearcoatGloss * 0.001f };
	Float Dr{ gtr1(absCosTheta(H), alphaG) };
	Float Fr{ frSchlick(.04f, dot(V, H)) };
	Float Gr{ smithG(absCosTheta(V), .25f) * smithG(absCosTheta(L), .25f) };

	return mat.clearcoat * Gr * Fr * Dr / 4.0f;
}


DEVICE Float pdfDisneyClearcoat(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	if (!sameHemisphere(V, L)) return 0.0f;

	Float3 H{ L + V };
	if (H.x == 0 && H.y == 0 && H.z == 0) return 0.0f;
	H = normalize(H);

	// The sampling routine samples H exactly from the GTR1 distribution.
	// Thus, the final value of the PDF is just the value of the
	// distribution for H converted to a measure with respect to the
	// surface normal.
	// - pbrt book v3
	Float alphaG{ (1 - mat.clearcoatGloss) * 0.1f + mat.clearcoatGloss * 0.001f };
	Float Dr{ gtr1(absCosTheta(H), alphaG) };
	return Dr * absCosTheta(H) / (4.0f * absCosTheta(V));
}


DEVICE void sampleDisneyClearcoat(MaterialStruct const& mat, Float3 const& V, Float3& L, 
	Random& rand, Float3& bsdf, Float& pdf)
	// there is no visible normal sampling for BRDF because the Clearcoat has no
	// physical meaning
	// Apply importance sampling to D * dot(N * H) / (4 * dot(H, V)) by choosing normal
	// proportional to D and reflect it at H
{
	Float alphaG{ (1 - mat.clearcoatGloss) * 0.1f + mat.clearcoatGloss * 0.001f };
	Float alpha2{ alphaG * alphaG };
	Float cosTheta{ sqrtf(max(0.0f, (1 - powf(alpha2, 1 - rand.random())) / (1 - alpha2))) };
	Float sinTheta{ sqrtf(max(0.0f, 1 - cosTheta * cosTheta)) };
	Float phi{ TWO_PI * rand.random() };

	Float3 H{ toSphereCoordinates(sinTheta, cosTheta, phi) };

	if (!sameHemisphere(V, H)) H = -H;

	bsdf = Float3{ 0.0f };
	pdf  = 0.0f;

	L = reflect(V, H);
	if (!sameHemisphere(V, L)) return;

	pdf = pdfDisneyClearcoat(mat, V, L);
	bsdf = fDisneyClearcoat(mat, V, L);
}

#pragma endregion

// ──────────────────────────────────────────────────────────────────────────────────────

#pragma region MICROFACETS

DEVICE Float3 fDisneyMicrofacet(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	if (!sameHemisphere(V, L)) return Float3{ 0.0f };
	Float NdotV{ absCosTheta(V) };
	if (NdotV == 0) return Float3{ 0.0f };

	Float3 H{ normalize(V + L) };

	Float2 alpha{ roughnessToAlpha(mat.roughness, mat.anisotropic) };
	Float Dr{ gtr2(H, alpha.x, alpha.y) };
	Float3 Fr{ mix(mat.baseColor, 1.0f - mat.baseColor, schlickWeight(dot(H, V)))};
	Float Gr{ smithGAnisotropic(absCosTheta(V), alpha) * 
		smithGAnisotropic(absCosTheta(L), alpha) };

	return Dr * Fr * Gr / (4.0f * absCosTheta(L));
}


DEVICE Float pdfDisneyMicrofacet(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	if (!sameHemisphere(V, L)) return 0.0f;

	Float3 H{ L + V };
	if (H.x == 0 && H.y == 0 && H.z == 0) return 0.0f;
	H = normalize(H);

	Float2 alpha{ roughnessToAlpha(mat.roughness, mat.anisotropic) };
	return pdfGtr2(V, H, alpha.x, alpha.y);
}


DEVICE void sampleDisneyMicrofacet(MaterialStruct const& mat, Float3 const& V, Float3& L,
	Random& rand, Float3& bsdf, Float& pdf)
{
	Float2 alpha{ roughnessToAlpha(mat.roughness, mat.anisotropic) };
	Float3 H{ sampleGtr2(V, alpha.x, alpha.y, {rand.random(), rand.random()}) };

	L = reflect(V, H);

	bsdf = Float3{ 0.0f };
	pdf = 0.0f;

	if (!sameHemisphere(V, L)) return;

	bsdf = fDisneyMicrofacet(mat, V, L);
	pdf = pdfDisneyMicrofacet(mat, V, L);
}

#pragma endregion

// ──────────────────────────────────────────────────────────────────────────────────────

#pragma region "MICROFACETS TRANSMISSION"

DEVICE Float3 fDisneyMicrofacetTransmission(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{

}


DEVICE Float pdfDisneyMicrofacetTransmission(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{

}


DEVICE void sampleDisneyMicrofacetTransmission(MaterialStruct const& mat, Float3 const& V, Float3& L,
	Random& rand, Float3& bsdf, Float& pdf)
{

}

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           DISNEY'S PRINCIPLED BSDF
//
//  Specular BSDF ───────┐ Metallic BRDF ┐
//                       │               │
//  Dielectric BRDF ─── MIX ─────────── MIX ─── DISNEY BSDF  
//  + Subsurface
//



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
