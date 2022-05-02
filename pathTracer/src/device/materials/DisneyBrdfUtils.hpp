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

DEVICE_INL Float schlickWeight(Float cosTheta)
{
	Float m{ owl::clamp(1.0f - cosTheta, 0.0f, 1.0f) };
	Float m2{ m * m };
	return m2 * m2 * m; // pow(m,5)
}

DEVICE_INL Float frSchlick(Float R0, Float cosTheta)
{
	return mix(1.0f, schlickWeight(cosTheta), R0);
}

DEVICE_INL Float3 frSchlick(Float3 R0, Float cosTheta)
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

DEVICE_INL bool relativeIOR(Float3 const& V, Float IOR, Float& inEta, Float& extEta)
{
	bool entering{ cosTheta(V) > 0.0f };
	inEta = entering ? IOR : 1.0f;
	extEta = entering ? 1.0f : IOR;
	return entering;
}

DEVICE_INL Float disneyFresnel(MaterialStruct const& mat, Float3 const& V,
	Float3 const& L, Float3 const& H)
{
	Float Fmetallic{ schlickWeight(dot(L,H)) };
	Float Fdielectric{ dielectricFresnel(abs(dot(V,H)), mat.ior) };
	return mix(Fdielectric, Fmetallic, mat.metallic);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  SMITH SHADOWING TERM
//  < Microfacet Models for Refraction through Rough Surfaces - Bruce Walter
//  > term is for single-scattering accurate and correct
//	> newer version which got adopted in 2014 revision
//  > NOTE: this term is not energy conserving for multi-scattering events

DEVICE_INL Float smithG(Float cosTheta, Float alpha)
{
	Float alpha2{ alpha * alpha };
	Float cosTheta2{ cosTheta * cosTheta };
	return 1.0f / (cosTheta + sqrtf(alpha2 + cosTheta2 - alpha2 * cosTheta2));
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
	Float alpha2{ alpha * alpha };
	Float t{ 1.0f + (alpha2 - 1.0f) * cosThetaH * cosThetaH };
	return (alpha2 - 1.0f) / (PI * logf(alpha2) * t);
}

//  GTR2 - notes eq. 8 (know as GGX)
DEVICE Float gtr2(Float3 const& H, Float alphax, Float alphay) // D
{
	Float tanTheta2{ tan2Theta(H) };
	if (isnan(tanTheta2)) return 0.0f;
	Float cos4Theta{ cos2Theta(H) * cos2Theta(H) };
	Float e{ (cos2Phi(H) / (alphax * alphax) + 
		sin2Phi(H) / (alphay * alphay)) * tanTheta2 };
	return 1 / (PI * alphax * alphay * cos4Theta * (1 + e) * (1 + e));
}

DEVICE Float gtr2Lambda(Float3 const& H, Float alphax, Float alphay)
{
	Float absTanTheta{ abs(tanTheta(H)) };
	if (isinf(absTanTheta)) return 0.;
	Float alpha{ sqrtf(cos2Phi(H) * alphax * alphax + sin2Phi(H) * alphay * alphay) };
	Float alpha2Tan2Theta{ (alpha * absTanTheta) * (alpha * absTanTheta) };
	return (-1 + sqrtf(1.f + alpha2Tan2Theta)) / 2;
}

DEVICE Float3 sampleGtr2(Float3 const& V, Float alphax, Float alphay, Float2 u)
{
	Float3 H{};

	Float cosTheta = 0, phi = (2 * PI) * u[1];
	if (alphax == alphay) {
		Float tanTheta2 = alphax * alphax * u[0] / (1.0f - u[0]);
		cosTheta = 1 / sqrtf(1 + tanTheta2);
	}
	else {
		phi = atanf(alphay / alphax * tanf(2 * PI * u[1] + .5f * PI));

		if (u[1] > .5f) phi += PI;
		Float sinPhi = sinf(phi), cosPhi = cosf(phi);
		Float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
		Float alpha2 = 1 / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
		Float tanTheta2 = alpha2 * u[0] / (1 - u[0]);
		cosTheta = 1 / sqrtf(1 + tanTheta2);
	}
	Float sinTheta = sqrtf(max((Float)0., (Float)1. - cosTheta * cosTheta));
	H = toSphereCoordinates(sinTheta, cosTheta, phi);
	if (!sameHemisphere(V, H)) H = -H;

	return H;
}

DEVICE Float pdfGtr2(Float3 const& V, Float3 const& H, Float alphax, Float alphay)
{
	Float Dr{ gtr2(H, alphax, alphay) };
	Float Gr{ smithGAnisotropic(V, {alphax, alphay}) };
	return Dr * absCosTheta(H) / (4.0f * absCosTheta(V));
}






#endif // !DISNEY_BRDF_UITLS_HPP
