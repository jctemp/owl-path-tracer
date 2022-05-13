#ifndef DISNEY_BRDF_HPP
#define DISNEY_BRDF_HPP
#pragma once

#include "../Sampling.hpp"
#include "BrdfUtils.hpp"

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

	Float FL{ schlickFresnel(NdotL, mat.ior) };
	Float FV{ schlickFresnel(NdotV, mat.ior) };

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
	Float FL{ schlickFresnel(NdotL, mat.ior) };
	Float FV{ schlickFresnel(NdotV, mat.ior) };
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

	Float FL{ schlickFresnel(NdotL, mat.ior) };
	Float FV{ schlickFresnel(NdotV, mat.ior) };
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
	return mat.sheen * mix(Float3{ 1.0f }, tint, Float3{ mat.sheenTint }) * schlickFresnel(cosThetaD, mat.ior);
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
	Float F{ schlickFresnel(absCosTheta(H), 1.5f) };
	Float Fr{ mix(.04f, 1.0f, F) };
	Float Gr{ smithG(absCosTheta(V), .25f) * smithG(absCosTheta(L), .25f) };

	return mat.clearcoat * Gr * Fr * Dr / (4.0f * absCosTheta(H));
}


DEVICE Float pdfDisneyClearcoat(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
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
	pdf = 0.0f;

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

	Float alpha{ roughnessToAlpha(mat.roughness) };
	Float Dr{ gtr2(cosTheta(H), alpha) };
	Float3 Fr{ mix(mat.baseColor, 1.0f - mat.baseColor, {schlickFresnel(dot(H, V), mat.ior)}) };
	Float Gr{ smithG(abs(tanTheta(V)), alpha) *
		smithG(abs(tanTheta(L)), alpha) };

	return Dr * Fr * Gr / (4.0f * absCosTheta(L));
}


DEVICE Float pdfDisneyMicrofacet(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	if (!sameHemisphere(V, L)) return 0.0f;

	Float3 H{ L + V };
	if (H.x == 0 && H.y == 0 && H.z == 0) return 0.0f;
	H = normalize(H);

	Float alpha{ roughnessToAlpha(mat.roughness) };
	return pdfGtr2(V, H, alpha);
}


DEVICE void sampleDisneyMicrofacet(MaterialStruct const& mat, Float3 const& V, Float3& L,
	Random& rand, Float3& bsdf, Float& pdf)
{
	Float alpha{ roughnessToAlpha(mat.roughness) };
	Float3 H{ sampleGtr2VNDF(V, alpha, {rand.random(), rand.random()}) };

	L = reflect(V, H);

	bsdf = Float3{ 0.0f };
	pdf = 0.0f;

	if (!sameHemisphere(V, L)) return;

	bsdf = fDisneyMicrofacet(mat, V, L);
	pdf = pdfDisneyMicrofacet(mat, V, L);
}

#pragma endregion

// ──────────────────────────────────────────────────────────────────────────────────────

#pragma region TRANSMISSION

DEVICE Float3 fDisneyMicrofacetTransmission(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	Float ior = mat.ior;
	bool entering = cosTheta(L) > 0;
	Float etaI = entering ? ior : 1;
	Float etaT = entering ? 1 : ior;
	Float eta = etaI / etaT;

	Float alpha{ roughnessToAlpha(mat.roughness) };

	// Microfacet models for refraction eq. (16)
	Float3 H{ L + V * eta };
	if (H.x == 0 && H.y == 0 && H.z == 0) return 0.0f;
	H = normalize(H);

	if (H.z < 0) H = -H;

	Float Dr{ gtr2(cosTheta(H), alpha) };

	return Dr * mat.transmission * (1.0f - mat.metallic);
}


DEVICE Float pdfDisneyMicrofacetTransmission(MaterialStruct const& mat, Float3 const& V, Float3 const& L)
{
	Float ior = mat.ior;
	bool entering = cosTheta(L) > 0;
	Float etaI = entering ? ior : 1;
	Float etaT = entering ? 1 : ior;
	Float eta = etaI / etaT;

	// Microfacet models for refraction eq. (16)
	Float3 H{ L + V * eta };
	if (H.x == 0 && H.y == 0 && H.z == 0) return 0.0f;
	H = normalize(H);

	Float alpha{ roughnessToAlpha(mat.roughness) };
	Float costhetaH = absCosTheta(H);
	Float D = gtr2(costhetaH, alpha);
	return D;
}

/*if (mat.transmission > 0.0f)
//{
//	return;
//	Float ior = mat.ior;
//	bool entering = cosTheta(L) > 0;
//	Float etaI = entering ? ior : 1;
//	Float etaT = entering ? 1 : ior;
//	Float3 N{ cosTheta(V) > 0 ? Float3{ 0,0,1 } : Float3{ 0, 0, -1 } };
//	bool refracted = refract(V, N, etaI / etaT, L);
//	Float fresnel = 1 - dielectricFresnel(absCosTheta(L), etaI / etaT);
//	if (!refracted || rand.random() > fresnel)
//	{
//		L = reflect(V, N);
//		return;
//	}
//	bsdf = Float3{ 1.0f };
//	pdf = 1;
//	return;
//}*/

DEVICE void sampleDisneyMicrofacetTransmission(MaterialStruct const& mat, Float3 const& V, Float3& L,
	Random& rand, Float3& bsdf, Float& pdf)
{
	if (mat.roughness < .2f) // check smoothness
	{
		Float ior = mat.ior;
		bool entering = cosTheta(L) > 0;
		Float etaI = entering ? ior : 1;
		Float etaT = entering ? 1 : ior;

		// Sample perfectly specular dielectric BSDF
		Float R{ schlickFresnel(cosTheta(V), etaI / etaT) };

		Float3 N{ cosTheta(V) > 0 ? Float3{0,0,1} : Float3{0,0,-1} };
		bsdf = Float3{ 0.0f };
		pdf = 0.0f;

		 //reflection => NO IDEA WHY THIS WORKS
		if (rand.random() < R * rand.random())
		{
			L = reflect(V, N);
			bsdf = Float3{ 1.0f };
			pdf = 1.0f;
			return;
		}

		// refraction
		bool refracted = refract(V, N, etaI / etaT, L);
		if (!refracted) return;


		bsdf = Float3{ 1.0f };
		pdf = 1.0f;

		return;
	}

	bsdf = Float3{ 0.0f };
	pdf = 0.0f;

}

// ──────────────────────────────────────────────────────────────────────────────────────
#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           DISNEY'S PRINCIPLED BSDF
//
//  Specular BSDF ───────┐ Metallic BRDF ┐
//                       │               │
//  Dielectric BRDF ─── MIX ─────────── MIX ─── DISNEY BSDF  
//  + Subsurface
//

DEVICE void sampleDisneyBSDF(MaterialStruct const& mat, Float3 const& V, Float3& L,
	Random& rand, Float3& bsdf, Float& pdf)
{
	bsdf = Float3{ 0.0f };
	pdf = 0.0f;

	Float r1 = rand.random(), r2 = rand.random();

	Float specularWeight = mat.metallic;
	Float diffuseWeight = 1 - mat.metallic;
	Float clearcoatWeight = 1.0f * saturate<Float>(mat.clearcoat);

	Float norm = 1.0f / (clearcoatWeight + diffuseWeight + specularWeight);

	Float pSpecular = specularWeight * norm;
	Float pDiffuse = diffuseWeight * norm;
	Float pClearcoat = clearcoatWeight * norm;

	Float cdf[3]{};
	cdf[0] = pDiffuse;
	cdf[1] = pSpecular + cdf[0];
	cdf[2] = pClearcoat + cdf[1];


	// diffuse
	if (r1 < cdf[0])
	{
		Float3 H{ normalize(L + V) };

		if (H.z < 0.0f) H = -H;

		sampleCosineHemisphere({ rand.random(), rand.random() }, L);
		pdfCosineHemisphere(V, L, pdf);

		auto diffuse = fDisneyDiffuse(mat, V, L);
		auto ss = fDisneyFakeSubsurface(mat, V, L);
		auto retro = fDisneyRetro(mat, V, L);
		auto sheen = fDisneySheen(mat, V, L);

		auto diffuseWeight = 1 - mat.subsurface;
		auto ssWeight = mat.subsurface;

		bsdf += (diffuse + retro) * diffuseWeight;
		bsdf += ss * ssWeight;
		bsdf += sheen;
		bsdf *= absCosTheta(L);
	}

	// specular reflective
	else if (r1 < cdf[1])
	{
		Float alpha{ roughnessToAlpha(mat.roughness) };
		Float3 H{ sampleGtr2VNDF(V, alpha, {rand.random(), rand.random()}) };

		L = normalize(reflect(V, H));

		if (!sameHemisphere(V, L)) return;

		bsdf = fDisneyMicrofacet(mat, V, L);
		pdf = pdfDisneyMicrofacet(mat, V, L);
	}

	// clearcoat
	else
	{
		sampleDisneyClearcoat(mat, V, L, rand, bsdf, pdf);
	}

}

#endif // !DISNEY_BRDF_HPP
