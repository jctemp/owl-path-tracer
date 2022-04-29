#ifndef DISNEY_BRDF_UITLS_HPP
#define DISNEY_BRDF_UITLS_HPP
#pragma once

#include "../Sampling.hpp"

DEVICE_INL Float luminance(Float3 color)
{
	return 0.212671f * color.x + 0.715160f * color.y + 0.072169f * color.z;
}

DEVICE_INL Float schlickFresnel(Float cosTheta)
{
	return pow(saturate<Float>(1.f - cosTheta), 5.f);
}

DEVICE_INL Float smith_shadowing_ggx(Float n_dot_o, Float alpha_g) {
	Float a = alpha_g * alpha_g;
	Float b = n_dot_o * n_dot_o;
	return 1.f / (n_dot_o + sqrt(a + b - a * b));
}

DEVICE_INL bool relativeIOR(Float3 const& V, Float3 const& N, Float const& ior,
	Float& etaO, Float& etaI)
{
	bool entering{ dot(V, N) > 0.f };
	etaI = entering ? 1.f : ior;
	etaO = entering ? ior : 1.f;
	return entering;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           GENERALIZED TROWBRIDGE-REITZ (gamma=1)
//           > Burley notes eq. 4


DEVICE Float GTR1(Float const& cosThetaH, Float const& alpha) {
	if (alpha >= 1.f) {
		return INV_PI;
	}

	Float alpha2{ alpha * alpha };
	return INV_PI * (alpha2 - 1.f) / (log(alpha2) * (1.f + (alpha2 - 1.f)
		* cosThetaH * cosThetaH));
}


DEVICE void pdfGTR1(Float3 const& V, Float3 const& L, Float3 const& H, Float3 const& N,
	Float const& alpha, Float& clearcoat)
{
	if (!sameHemisphere(V, L, N))
	{
		clearcoat = 0.0f;
		return;
	}

	Float cosThetaH{ dot(N, H) };
	Float d{ GTR1(cosThetaH, alpha) };
	clearcoat = d * cosThetaH / (4.f * dot(V, H));
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           GENERALIZED TROWBRIDGE-REITZ (gamma=2)
//           > Burley notes eq. 8


DEVICE Float GTR2(Float const& cosThetaH, Float const& alpha)
{
	Float alpha2{ alpha * alpha };
	return INV_PI * alpha2 / fmaxf(pow2(1.f + (alpha2 - 1.f) * cosThetaH *
		cosThetaH), EPSILON);
}


DEVICE void pdfGTR2(Float3 const& V, Float3 const& L, Float3 const& H, Float3 const& N,
	Float const& alpha, Float& microfacet)
{
	if (!sameHemisphere(V, L, N))
	{
		microfacet = 0.0f;
		return;
	}

	Float cosThetaH{ dot(N, H) };
	Float d{ GTR2(cosThetaH, alpha) };
	microfacet = d * cosThetaH / (4.f * fabs(dot(V, H)));
}


DEVICE void pdfGTR2Transmission(Float3 const& V, Float3 const& L, Float3 const& N,
	Float const& transmissionRoughness, Float const& ior, Float& microfacetTransmission)
{
	if (sameHemisphere(V, L, N))
	{
		microfacetTransmission = 0.0f;
		return;
	}

	Float alpha{ fmaxf(0.001f, transmissionRoughness * transmissionRoughness) };
	Float etaO, etaI;
	bool entering = relativeIOR(V, N, ior, etaO, etaI);

	Float3 Ht{ normalize(-(V * etaI + L * etaO)) };
	Float cosThetaH{ fabs(dot(N, Ht)) };
	Float d{ GTR2(cosThetaH, alpha) };
	microfacetTransmission = d;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           GENERALIZED TROWBRIDGE-REITZ (gamma=2)
//           > Burley notes eq. 8


#endif // !DISNEY_BRDF_UITLS_HPP
