#ifndef DISNEY_BRDF_UITLS_HPP
#define DISNEY_BRDF_UITLS_HPP
#pragma once

#include "../Sampling.hpp"

// SOURCES
// https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf
// https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.2297&rep=rep1&type=pdf
// https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf



DEVICE_INL Float luminance(Float3 color)
{
	return 0.212671f * color.x + 0.715160f * color.y + 0.072169f * color.z;
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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  SMITH SHADOWING GGX
//  < Microfacet Models for Refraction through Rough Surfaces - Bruce Walter
//  > term is for single-scattering accurate and correct
//  > NOTE: this term is not energy conserving for multi-scattering events

DEVICE_INL Float smithGAnisotropic(Float NdotV, Float VdotX, Float VdotY, Float2 alpha)
{
	return 1.f / (NdotV + sqrtf(pow2(VdotX * alpha.x) + pow2(VdotY * alpha.y) + pow2(NdotV)));
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  GENERALIZED TROWBRIDGE-REITZ DISTRIBUTION
//  < Disney BRDF notes 2012 - Burley
//  > has long tails and short peaks in the distribution curve
//  > allows for normalisation and importance sampling
//  > Disney uses two fix specular lobes with gamma = [1, 2]
//  > alpha = roughness^2 result in better linearity

//  GTR1 - notes eq. 4
DEVICE_INL Float gtr1(Float cosThetaH, Float alpha)
{
	Float alpha2{ alpha * alpha };
	return INV_PI * (alpha2 - 1.0f) / (logf(alpha2) * (1.0f + (alpha2 - 1.0f) * pow2(cosThetaH)));
}

//  GTR2 - notes eq. 8
DEVICE_INL Float gtr2(Float cosThetaH, Float alpha)
{
	Float alpha2{ alpha * alpha };
	return INV_PI * alpha2 / max(pow2(1 + (alpha2 * alpha2 - 1) * pow2(cosThetaH)), MIN_ALPHA);
}

//  GTR3
DEVICE_INL Float gtr2Anisotropic(Float NdotH, Float HdotX, Float HdotY, Float2 alpha)
{
	return INV_PI / max(alpha.x * alpha.y * pow2(pow2(HdotX / alpha.x) +
		pow2(HdotY / alpha.y) + pow2(NdotH)), MIN_ALPHA);
}

//	SAMPLING MICROFACET NORMAL H





#endif // !DISNEY_BRDF_UITLS_HPP
