#ifndef DISNEY_BRDF_HPP
#define DISNEY_BRDF_HPP
#pragma once

#include "../../sample_methods.hpp"
#include "BrdfUtils.hpp"
#include "../../types.hpp"

// REFECRENCES:
// https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           disney'S COMPONENTS

#pragma region DIFFUSE

__device__ vec3 fDisneyDiffuse(material_data const& mat, vec3 const& V, vec3 const& L)
{
	float NdotV{ owl::abs(cos_theta(V)) };
	float NdotL{ owl::abs(cos_theta(L)) };

	float FL{ schlickFresnel(NdotL, mat.ior) };
	float FV{ schlickFresnel(NdotV, mat.ior) };

	// Burley 2015, eq (4).
	return mat.base_color * inv_pi * (1.0f - FL / 2.0f) * (1.0f - FV / 2.0f);
}


__device__ float pdfDisneyDiffuse(material_data const& mat, vec3 const& V, vec3 const& L)
{
	return pdf_cosine_hemisphere(V, L);
}


__device__ void sampleDisneyDiffuse(material_data const& mat, vec3 const& V, vec3& L,
	Random& rand, vec3& bsdf, float& pdf)
{
    L = sample_cosine_hemisphere({rand.random(), rand.random()});
	pdf = pdfDisneyDiffuse(mat, V, L);
	bsdf = fDisneyDiffuse(mat, V, L) * owl::abs(cos_theta(L));
}

#pragma endregion

// ──────────────────────────────────────────────────────────────────────────────────────

#pragma region "FAKE SUBSURFACE"

__device__ vec3 fDisneyFakeSubsurface(material_data const& mat, vec3 const& V, vec3 const& L)
// Hanrahan - Krueger BRDF approximation of the BSSRDF
{
	vec3 H{ L + V };
	if (H.x == 0 && H.y == 0 && H.z == 0) return vec3{ 0.0f };

	H = normalize(H);
	float cosThetaD{ dot(L, H) };
	float NdotV{ owl::abs(cos_theta(V)) };
	float NdotL{ owl::abs(cos_theta(L)) };

	// Fss90 used to "flatten" retroreflection based on roughness
	float Fss90{ cosThetaD * cosThetaD * mat.roughness };
	float FL{ schlickFresnel(NdotL, mat.ior) };
	float FV{ schlickFresnel(NdotV, mat.ior) };
	float Fss{ lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV) };

	// 1.25 scale is used to (roughly) preserve albedo
	float Fs{ 1.25f * (Fss * (1.0f / (NdotV + NdotL) - 0.5f) + 0.5f) };

	return mat.subsurface_color * inv_pi * Fs;
}


__device__ float pdfDisneyFakeSubsurface(material_data const& mat, vec3 const& V, vec3 const& L)
{
	return pdf_cosine_hemisphere(V, L);
}


__device__ void sampleDisneyFakeSubsurface(material_data const& mat, vec3 const& V, vec3& L,
	Random& rand, vec3& bsdf, float& pdf)
{
    L = sample_cosine_hemisphere({rand.random(), rand.random()});
	pdf = pdfDisneyFakeSubsurface(mat, V, L);
	bsdf = fDisneyFakeSubsurface(mat, V, L) * owl::abs(cos_theta(L));
}

#pragma endregion

// ──────────────────────────────────────────────────────────────────────────────────────

#pragma region RETRO

__device__ vec3 fDisneyRetro(material_data const& mat, vec3 const& V, vec3 const& L)
{
	vec3 H{ L + V };
	if (H.x == 0 && H.y == 0 && H.z == 0) return vec3{ 0.0f };

	H = normalize(H);
	float cosThetaD{ dot(L, H) };
	float NdotV{ owl::abs(cos_theta(V)) };
	float NdotL{ owl::abs(cos_theta(L)) };

	float FL{ schlickFresnel(NdotL, mat.ior) };
	float FV{ schlickFresnel(NdotV, mat.ior) };
	float Rr{ 2 * mat.roughness * cosThetaD * cosThetaD };

	// Burley 2015, eq (4).
	return mat.base_color * inv_pi * Rr * (FL + FV + FL * FV * (Rr - 1));
}


__device__ float pdfDisneyRetro(material_data const& mat, vec3 const& V, vec3 const& L)
{
	return pdf_cosine_hemisphere(V, L);
}


__device__ void sampleDisneyRetro(material_data const& mat, vec3 const& V, vec3& L,
	Random& rand, vec3& bsdf, float& pdf)
{
    L = sample_cosine_hemisphere({rand.random(), rand.random()});
	pdf = pdfDisneyRetro(mat, V, L);
	bsdf = fDisneyRetro(mat, V, L) * owl::abs(cos_theta(L));
}

#pragma endregion

// ──────────────────────────────────────────────────────────────────────────────────────

#pragma region SHEEN

__device__ vec3 fDisneySheen(material_data const& mat, vec3 const& V, vec3 const& L)
{
	vec3 H{ L + V };
	if (H.x == 0 && H.y == 0 && H.z == 0) return vec3{ 0.0f };
	if (mat.sheen <= 0.0f) return vec3{ 0.0f };

	H = normalize(H);
	float cosThetaD{ dot(L, H) };

	vec3 tint{ calculateTint(mat.base_color) };
	return mat.sheen * lerp(vec3{ 1.0f }, tint, vec3{ mat.sheen_tint }) * schlickFresnel(cosThetaD, mat.ior);
}


__device__ float pdfDisneySheen(material_data const& mat, vec3 const& V, vec3 const& L)
{
	return pdf_cosine_hemisphere(V, L);
}


__device__ void sampleDisneySheen(material_data const& mat, vec3 const& V, vec3& L,
	Random& rand, vec3& bsdf, float& pdf)
{
    L = sample_cosine_hemisphere({rand.random(), rand.random()});
	pdf = pdfDisneySheen(mat, V, L);
	bsdf = fDisneySheen(mat, V, L) * owl::abs(cos_theta(L));
}

#pragma endregion

// ──────────────────────────────────────────────────────────────────────────────────────

#pragma region CLEARCOAT

__device__ vec3 fDisneyClearcoat(material_data const& mat, vec3 const& V, vec3 const& L)
{
	vec3 H{ L + V };
	if (H.x == 0 && H.y == 0 && H.z == 0) return vec3{ 0.0f };
	H = normalize(H);

	// Clearcoat has ior = 1.5 hardcoded -> F0 = 0.04. It then uses the
	// GTR1 distribution, which has even fatter tails than Trowbridge-Reitz
	// (which is GTR2). The geometric term always based on alpha = 0.25
	// - pbrt book v3
	float alphaG{(1 - mat.clearcoat_gloss) * 0.1f + mat.clearcoat_gloss * 0.001f };
	float Dr{ gtr1(owl::abs(cos_theta(H)), alphaG) };
	float F{ schlickFresnel(owl::abs(cos_theta(H)), 1.5f) };
	float Fr{ lerp(.04f, 1.0f, F) };
	float Gr{ smithG(owl::abs(cos_theta(V)), .25f) * smithG(owl::abs(cos_theta(L)), .25f) };

	return mat.clearcoat * Gr * Fr * Dr / (4.0f * owl::abs(cos_theta(H)));
}


__device__ float pdfDisneyClearcoat(material_data const& mat, vec3 const& V, vec3 const& L)
{
	vec3 H{ L + V };
	if (H.x == 0 && H.y == 0 && H.z == 0) return 0.0f;
	H = normalize(H);

	// The sampling routine samples H exactly from the GTR1 distribution.
	// Thus, the final value of the PDF is just the value of the
	// distribution for H converted to a measure with respect to the
	// surface normal.
	// - pbrt book v3
	float alphaG{(1 - mat.clearcoat_gloss) * 0.1f + mat.clearcoat_gloss * 0.001f };
	float Dr{ gtr1(owl::abs(cos_theta(H)), alphaG) };
	return Dr * owl::abs(cos_theta(H)) / (4.0f * owl::abs(cos_theta(V)));
}


__device__ void sampleDisneyClearcoat(material_data const& mat, vec3 const& V, vec3& L,
	Random& rand, vec3& bsdf, float& pdf)
	// there is no visible normal sampling for BRDF because the Clearcoat has no
	// physical meaning
	// Apply importance sampling to D * dot(N * H) / (4 * dot(H, V)) by choosing normal
	// proportional to D and reflect it at H
{
	float alphaG{(1 - mat.clearcoat_gloss) * 0.1f + mat.clearcoat_gloss * 0.001f };
	float alpha2{ alphaG * alphaG };
	float cosTheta{ sqrtf(fmax(0.0f, (1 - powf(alpha2, 1 - rand.random())) / (1 - alpha2))) };
	float sinTheta{ sqrtf(fmax(0.0f, 1 - cosTheta * cosTheta)) };
	float phi{ two_pi * rand.random() };

	vec3 H{to_sphere_coordinates(sinTheta, cosTheta, phi) };

	if (!same_hemisphere(V, H)) H = -H;

	bsdf = vec3{ 0.0f };
	pdf = 0.0f;

	L = reflect(V, H);
	if (!same_hemisphere(V, L)) return;

	pdf = pdfDisneyClearcoat(mat, V, L);
	bsdf = fDisneyClearcoat(mat, V, L);
}

#pragma endregion

// ──────────────────────────────────────────────────────────────────────────────────────

#pragma region MICROFACETS

__device__ vec3 fDisneyMicrofacet(material_data const& mat, vec3 const& V, vec3 const& L)
{
	if (!same_hemisphere(V, L)) return vec3{0.0f };
	float NdotV{ owl::abs(cos_theta(V)) };
	if (NdotV == 0) return vec3{ 0.0f };

	vec3 H{ normalize(V + L) };

	float alpha{ roughnessToAlpha(mat.roughness) };
	float Dr{ gtr2(cos_theta(H), alpha) };
	vec3 Fr{ lerp(mat.base_color, 1.0f - mat.base_color, {schlickFresnel(dot(H, V), mat.ior)}) };
	float Gr{smithG(abs(tan_theta(V)), alpha) *
             smithG(abs(tan_theta(L)), alpha) };

	return Dr * Fr * Gr / (4.0f * owl::abs(cos_theta(L)));
}


__device__ float pdfDisneyMicrofacet(material_data const& mat, vec3 const& V, vec3 const& L)
{
	if (!same_hemisphere(V, L)) return 0.0f;

	vec3 H{ L + V };
	if (H.x == 0 && H.y == 0 && H.z == 0) return 0.0f;
	H = normalize(H);

	float alpha{ roughnessToAlpha(mat.roughness) };
	return pdfGtr2(V, H, alpha);
}


__device__ void sampleDisneyMicrofacet(material_data const& mat, vec3 const& V, vec3& L,
	Random& rand, vec3& bsdf, float& pdf)
{
	float alpha{ roughnessToAlpha(mat.roughness) };
	vec3 H{ sampleGtr2VNDF(V, alpha, {rand.random(), rand.random()}) };

	L = reflect(V, H);

	bsdf = vec3{ 0.0f };
	pdf = 0.0f;

	if (!same_hemisphere(V, L)) return;

	bsdf = fDisneyMicrofacet(mat, V, L);
	pdf = pdfDisneyMicrofacet(mat, V, L);
}

#pragma endregion

// ──────────────────────────────────────────────────────────────────────────────────────
#ifdef TRANSMISSION_MATERIAL

#pragma region TRANSMISSION

__device__ vec3 fDisneyMicrofacetTransmission(MaterialStruct const& mat, vec3 const& V, vec3 const& L)
{
	float ior = mat.ior;
	bool entering = cosTheta(L) > 0;
	float etaI = entering ? ior : 1;
	float etaT = entering ? 1 : ior;
	float eta = etaI / etaT;

	float alpha{ roughnessToAlpha(mat.roughness) };

	// Microfacet models for refraction eq. (16)
	vec3 H{ L + V * eta };
	if (H.x == 0 && H.y == 0 && H.z == 0) return 0.0f;
	H = normalize(H);

	if (H.z < 0) H = -H;

	float Dr{ gtr2(cosTheta(H), alpha) };

	return Dr * mat.transmission * (1.0f - mat.metallic);
}


__device__ float pdfDisneyMicrofacetTransmission(MaterialStruct const& mat, vec3 const& V, vec3 const& L)
{
	float ior = mat.ior;
	bool entering = cosTheta(L) > 0;
	float etaI = entering ? ior : 1;
	float etaT = entering ? 1 : ior;
	float eta = etaI / etaT;

	// Microfacet models for refraction eq. (16)
	vec3 H{ L + V * eta };
	if (H.x == 0 && H.y == 0 && H.z == 0) return 0.0f;
	H = normalize(H);

	float alpha{ roughnessToAlpha(mat.roughness) };
	float costhetaH = owl::abs(cos_theta(H);
	float D = gtr2(costhetaH, alpha);
	return D;
}

/*if (mat.transmission > 0.0f)
//{
//	return;
//	float ior = mat.ior;
//	bool entering = cosTheta(L) > 0;
//	float etaI = entering ? ior : 1;
//	float etaT = entering ? 1 : ior;
//	Float3 N{ cosTheta(V) > 0 ? Float3{ 0,0,1 } : Float3{ 0, 0, -1 } };
//	bool refracted = refract(V, N, etaI / etaT, L);
//	float fresnel = 1 - dielectricFresnel(owl::abs(cos_theta(L), etaI / etaT);
//	if (!refracted || rand.random() > fresnel)
//	{
//		L = reflect(V, N);
//		return;
//	}
//	bsdf = Float3{ 1.0f };
//	pdf = 1;
//	return;
//}*/

__device__ void sampleDisneyMicrofacetTransmission(MaterialStruct const& mat, vec3 const& V, vec3& L,
	Random& rand, vec3& bsdf, float& pdf)
{
	if (mat.roughness < .2f) // check smoothness
	{
		float ior = mat.ior;
		bool entering = cosTheta(L) > 0;
		float etaI = entering ? ior : 1;
		float etaT = entering ? 1 : ior;

		// Sample perfectly specular dielectric BSDF
		float R{ schlickFresnel(cosTheta(V), etaI / etaT) };

		vec3 N{ cos_theta(V) > 0 ? vec3{0,0,1} : vec3{0,0,-1} };
		bsdf = vec3{ 0.0f };
		pdf = 0.0f;

		 //reflection => NO IDEA WHY THIS WORKS
		if (rand.random() < R * rand.random())
		{
			L = reflect(V, N);
			bsdf = vec3{ 1.0f };
			pdf = 1.0f;
			return;
		}

		// refraction
		bool refracted = refract(V, N, etaI / etaT, L);
		if (!refracted) return;


		bsdf = vec3{ 1.0f };
		pdf = 1.0f;

		return;
	}

	bsdf = vec3{ 0.0f };
	pdf = 0.0f;

}

// ──────────────────────────────────────────────────────────────────────────────────────
#pragma endregion

#endif // TRANSMISSION_MATERIAL

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           disney'S PRINCIPLED BSDF
//
//  Specular BSDF ───────┐ Metallic BRDF ┐
//                       │               │
//  Dielectric BRDF ─── lerp ─────────── lerp ─── disney BSDF
//  + Subsurface
//

__device__ void sampleDisneyBSDF(material_data const& mat, vec3 const& V, vec3& L,
	Random& rand, vec3& bsdf, float& pdf)
{
	bsdf = vec3{ 0.0f };
	pdf = 0.0f;

	float r1 = rand.random(), r2 = rand.random();

	float specularWeight = mat.metallic;
	float diffuseWeight = 1 - mat.metallic;
	float clearcoatWeight = 1.0f * o_saturate(mat.clearcoat);

	float norm = 1.0f / (clearcoatWeight + diffuseWeight + specularWeight);

	float pSpecular = specularWeight * norm;
	float pDiffuse = diffuseWeight * norm;
	float pClearcoat = clearcoatWeight * norm;

	float cdf[3]{};
	cdf[0] = pDiffuse;
	cdf[1] = pSpecular + cdf[0];
	cdf[2] = pClearcoat + cdf[1];


	// diffuse
	if (r1 < cdf[0])
	{
		vec3 H{ normalize(L + V) };

		if (H.z < 0.0f) H = -H;

        L = sample_cosine_hemisphere({rand.random(), rand.random()});
        pdf = pdf_cosine_hemisphere(V, L);

		auto diffuse = fDisneyDiffuse(mat, V, L);
		auto ss = fDisneyFakeSubsurface(mat, V, L);
		auto retro = fDisneyRetro(mat, V, L);
		auto sheen = fDisneySheen(mat, V, L);

		auto diffuseWeight = 1 - mat.subsurface;
		auto ssWeight = mat.subsurface;

		bsdf += (diffuse + retro) * diffuseWeight;
		bsdf += ss * ssWeight;
		bsdf += sheen;
		bsdf *= owl::abs(cos_theta(L));
	}

	// specular reflective
	else if (r1 < cdf[1])
	{
		float alpha{ roughnessToAlpha(mat.roughness) };
		vec3 H{ sampleGtr2VNDF(V, alpha, {rand.random(), rand.random()}) };

		L = normalize(reflect(V, H));

		if (!same_hemisphere(V, L)) return;

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
