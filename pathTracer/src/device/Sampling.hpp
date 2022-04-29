#ifndef SAMPLE_METHODS_HPP
#define SAMPLE_METHODS_HPP
#pragma once

#include "Globals.hpp"
#include "Math.hpp"

/*
* - http://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf.
* - https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
*/

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "GENERIC SAMPLING"

DEVICE void sampleUniformDisk(Float2& rand)
{
	float phi{ TWO_PI * rand.y };
	float r{ owl::sqrt(rand.x) };

	rand.x = r * owl::cos(phi);
	rand.y = r * owl::sin(phi);
}


DEVICE void sampleConcentricDisk(Float2& rand)
{
	// re-scale rand to be between [-1,1]
	float dx{ 2.0f * rand.x - 1 };
	float dy{ 2.0f * rand.y - 1 };

	// handle degenerated origin
	if (dx == 0 && dy == 0)
	{
		rand.x = 0;
		rand.y = 0;
		return;
	}

	// handle mapping unit squre to unit disk
	float phi, r;
	if (std::abs(dx) > std::abs(dy))
	{
		r = dx;
		phi = PI_OVER_FOUR * (dy / dx);
	}
	else
	{
		r = dy;
		phi = PI_OVER_TWO - PI_OVER_FOUR * (dx / dy);
	}

	rand.x = r * owl::cos(phi);
	rand.y = r * owl::sin(phi);
}


DEVICE Float3 sampleUniformSphere(Float2& rand)
{
	Float z{ 1.0f - 2.0f * rand.x };
	Float r{ sqrtf(fmaxf(0.0f, 1.0f - z * z)) };
	Float phi{ TWO_PI * rand.y };
	Float x = r * owl::cos(phi);
	Float y = r * owl::sin(phi);

	return Float3{ x, y, z };
}

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "COSINE HEMISPHERE SAMPLING"

DEVICE void sampleCosineHemisphere(Float2 rand, Float3& L)
{
	// 1. sample unit circle and save position into randu, randv
	sampleConcentricDisk(rand);

	// 2. calculate cosTheta => 1 = randu^2 + randv^2 => cos = 1 - (randu^2 + randv^2)
	Float cosTheta{ owl::sqrt(owl::max(0.0f, 1.0f - rand.x * rand.x - rand.y * rand.y)) };

	L = Float3{ rand.x, rand.y, cosTheta };
}


DEVICE void pdfCosineHemisphere(Float3 const& V, Float3 const& L, Float& pdf)
{
	pdf = absCosTheta(L) * INV_PI;
}

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "UNIFORM HEMISPHERE SAMPLING"

DEVICE void sampleUniformHemisphere(Float2 rand, Float3& L)
{
	Float z{ rand.x };
	Float r{ sqrtf(fmaxf(0.0f, 1.0f - z * z)) };
	Float phi = TWO_PI * rand.y;

	Float x = r * owl::cos(phi);
	Float y = r * owl::sin(phi);

	L = Float3{ x, y, z };
}


DEVICE void pdfUniformHemisphere(Float3 const& V, Float3 const& L, Float& pdf)
{
	pdf = 0.5f * INV_PI;
}

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "CONE SAMPLING"

DEVICE void sampleUniformCone(Float2 rand, Float cosThetaMax, Float3& w)
{
	Float cosTheta{ (1.0f - rand.u) + rand.u * cosThetaMax };
	Float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
	Float phi{ rand.v * 2.0f * PI };
	w = Float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
}


DEVICE void pdfUniformCone(Float cosThetaMax, Float& pdf)
{
	pdf = 1.0f / (2.0f * PI * (1.0f - cosThetaMax));
}

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



#endif // SAMPLE_METHODS_HPP

