#ifndef DISNEY_BRDF_UITLS_HPP
#define DISNEY_BRDF_UITLS_HPP
#pragma once

#include "../Sampling.hpp"

// SOURCES
// https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf
// https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.2297&rep=rep1&type=pdf
// https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
// https://jcgt.org/published/0007/04/01/

DEVICE_INL Float luminance(Float3 color)
{
	return 0.212671f * color.x + 0.715160f * color.y + 0.072169f * color.z;
}

DEVICE_INL Float3 calculateTint(Float3 baseColor)
// diseny uses in the BRDF explorer the luminance for the calculation
{
	Float lum{ luminance(baseColor) };
	return (lum > 0.0f) ? baseColor * (1.0f / lum) : Float3{ 1.0f };
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  FRESNEL TERM
//  < An Inexpensive BRDF Model for Physically-based Rendering - Christophe Schlick
//  > depending on needs the function can be tailored to more accurate calculations
//  > using here a very computational inexpensive solution

DEVICE_INL Float schlickFresnel(Float cosTheta)
{
	Float m{ owl::clamp(1.0f - cosTheta, 0.0f, 1.0f) };
	Float m2{ m * m };
	return m2 * m2 * m; // pow(m,5)
}

DEVICE_INL Float schlickFresnel(Float R0, Float cosTheta)
{
	return mix(1.0f, schlickFresnel(cosTheta), R0);
}

DEVICE_INL Float3 schlickFresnel(Float3 R0, Float cosTheta)
{
	Float3 exponential = powf(1.0f - cosTheta, 5.0f);
	return R0 + (Float3{ 1.0f } - R0) * exponential;
}

DEVICE_INL Float dielectricFresnel(Float cosThetaI, Float eta)
{
	Float sinThetaTSq{ eta * eta * (1.0f - cosThetaI * cosThetaI) };

	// Total internal reflection
	if (sinThetaTSq > 1.0f)
		return 1.0f;

	Float cosThetaT{ sqrtf(max(1.0f - sinThetaTSq, 0.0f)) };

	Float rs{ (eta * cosThetaT - cosThetaI) / (eta * cosThetaT + cosThetaI) };
	Float rp{ (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT) };

	return 0.5f * (rs * rs + rp * rp);
}

DEVICE_INL Float schlickR0FromRelativeIOR(Float eta)
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
{
	return pow2(eta - 1.0f) / pow2(eta + 1.0f);
}

DEVICE_INL Float disneyFresnel(MaterialStruct const& mat, Float3 const& V,
	Float3 const& L, Float3 const& H)
{
	Float Fmetallic{ schlickFresnel(dot(L,H)) };
	Float Fdielectric{ dielectricFresnel(abs(dot(V,H)), mat.ior) };
	return mix(Fdielectric, Fmetallic, mat.metallic);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  SMITH SHADOWING TERM
//  < Microfacet Models for Refraction through Rough Surfaces - Bruce Walter
//  > term is for single-scattering accurate and correct
//	> newer version which got adopted in 2014 revision
//  > NOTE: this term is not energy conserving for multi-scattering events

DEVICE_INL Float smithVCorrelated(Float NdotV, Float NdotL, Float alpha)
// https://google.github.io/filament/Filament.html#materialsystem/specularbrdf

{
	Float GGXV{ NdotL * sqrt(NdotV * NdotV * (1.0 - alpha) + alpha) };
	Float GGXL{ NdotV * sqrt(NdotL * NdotL * (1.0 - alpha) + alpha) };
	return 0.5f / (GGXV + GGXL);
}

DEVICE_INL Float smithG(Float NdotV, Float alpha)
{
	Float alpha2{ pow2(alpha) };
	Float dot2{ pow2(NdotV) };
	return (2.0f * NdotV) / (NdotV + sqrtf(alpha2 + dot2 - alpha2 * dot2));
}

DEVICE_INL Float smithGAnisotropic(Float3 const& V, Float2 alpha)
{
	Float tanTheta2{ tan2Theta(V) };
	if (isinf(tanTheta2)) return 0.0f;
	Float cosPhi2{ cos2Phi(V) };
	Float alpha2{ cosPhi2 * pow2(alpha.u) + (1.0f - cosPhi2) * pow2(alpha.v) };
	return 2.0f / (1.0f + sqrtf(1.0f + alpha2 * tanTheta2));
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  GENERALIZED TROWBRIDGE-REITZ DISTRIBUTION
//  < Disney BRDF notes 2012 - Burley
//  > has long tails and short peaks in the distribution curve
//  > allows for normalisation and importance sampling
//  > Disney uses two fix specular lobes with gamma = [1, 2]
//  > alpha = roughness^2 result in better linearity

DEVICE_INL Float roughnessToAlpha(Float roughness)
{
	return max(MIN_ALPHA, pow2(roughness));
}

DEVICE_INL Float2 roughnessToAlpha(Float roughness, Float anisotropic)
{
	Float aspect{ sqrtf(1.0f - 0.9f * anisotropic) };
	Float2 alpha{
		max(MIN_ALPHA, pow2(roughness) / aspect),
		max(MIN_ALPHA, pow2(roughness) * aspect)
	};
	return alpha;
}

//  GTR1 - notes eq. 4
DEVICE_INL Float gtr1(Float cosThetaH, Float alpha)
{
	if (alpha > 1.0f) return INV_PI;
	Float alpha2{ alpha * alpha };
	Float t{ 1.0 + (alpha2 - 1.0) * cosThetaH * cosThetaH };
	return (alpha2 - 1.0f) / (PI * logf(alpha2) * t);
}

//  GTR2 - notes eq. 8 (know as GGX)
DEVICE_INL Float gtr2(Float cosThetaH, Float alpha)
{
	Float alpha2{ pow2(alpha) };
	Float t{ 1.0f + (alpha2 - 1.0f) * pow2(cosThetaH) };
	return alpha2 / (PI * pow2(t));
}

//  GTR2 - later derived and used in 2015 Version
//	Anisotropic GGX (Trowbridge-Reitz) distribution formula, pbrt-v3 ( page 539 )
DEVICE_INL Float gtr2Anisotropic(Float3 H, Float2 alpha)
{
	Float cosThetaH2{ cos2Theta(H) };
	if (cosThetaH2 <= 0.0f) return 0.0f;
	Float beta{ cosThetaH2 + pow2(H.x) / pow2(alpha.x) + pow2(H.y) / pow2(alpha.y) };
	return 1.0f / (PI * alpha.x * alpha.y * pow2(beta));
}


DEVICE_INL Float pdfGtr2(Float3 const& V, Float3 const& L, Float3 const& H, Float alpha)
{
	if (!sameHemisphere(V, L))
		return 0.f;

	Float cosThetaH{ absCosTheta(H) };
	Float D{ gtr2(cosThetaH, alpha) };
	return D * cosThetaH / (4.f * fabsf(dot(V, H)));
}


DEVICE_INL Float pdfGtr2Anisotropic(Float3 const& V, Float3 const& L, Float3 const& H,
	Float2 alpha)
{
	printf("%f * %f = %f > 0.0f\n", L.z, V.z, L.z * V.z);
	if (!sameHemisphere(V, L))
		return 0.f;

	Float cosThetaH{ absCosTheta(H) };
	Float D{ gtr2Anisotropic(H, alpha) };
	return D * cosThetaH / (4.f * fabsf(dot(V, H)));
}


DEVICE_INL Float3 sampleGtr2Anisotropic(Float3 const& n, Float3 const& v_x, Float3 const& v_y,
	Float2 alpha, Float2 u) {
	Float x{ 2.f * M_PI * u.x };
	Float3 H = owl::sqrt(u.y / (1.0f - u.y)) * (alpha.x * cosf(x) * v_x + alpha.y * sinf(x) * v_y) + n;
	return normalize(H);
}


#endif // !DISNEY_BRDF_UITLS_HPP
