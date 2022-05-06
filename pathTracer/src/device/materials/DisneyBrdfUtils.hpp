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

DEVICE_INL Float calculateEta(Float3 V, Float ior)
{
	if (cosTheta(V) > 0.0f)
		return 1.0f / ior;
	return ior;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  FRESNEL TERM

DEVICE_INL Float schlickFresnel(Float costheta, Float ior)
{
	Float r0{ (1.0f - ior) / (1.0f + ior) };
	r0 = r0 * r0;
	Float m{ owl::clamp(1.0f - costheta, 0.0f, 1.0f) };
	Float m2{ m * m };
	return r0 + (1.0f - r0) * (m2 * m2 * m);
}

DEVICE_INL Float dielectricFresnel(Float costheta, Float ior)
{
	Float costhetaI{ costheta };

	if (costhetaI < 0.0f)
	{
		costhetaI = -costhetaI;
		ior = 1.0f / ior;
	}

	Float sin2thetaI{ 1.0f - costhetaI * costhetaI };
	Float sin2thetaT{ sin2thetaI / (ior * ior) };

	if (sin2thetaT > 1.0f)
	{
		return 1.0f;
	}

	Float cos2thetaT = sqrtf(1.0f - sin2thetaT);
	Float rp{ (costhetaI - ior * cos2thetaT) / (costhetaI + ior * cos2thetaT) };
	Float rs{ (ior * costhetaI - cos2thetaT) / (ior * costhetaI + cos2thetaT) };

	return 0.5f * (rp * rp + rs * rs);
}

DEVICE_INL Float dielectricFresnel(Float ior, Float3 V, Float3 N, Float3 & R, Float3& T, bool& inside)
{
	Float costheta{ cosTheta(V) }, neta{};
	Float3 Nn;

	if (costheta > 0)
	{
		neta = 1 / ior;
		inside = false;
		Nn = N;
	}
	else
	{
		neta = ior;
		inside = true;
		Nn = -N;
	}

	// compute reflection
	R = (2 * costheta) * Nn - V;

	Float arg = 1 - (neta * neta * (1 - (costheta * costheta)));
	if (arg < 0) 
	{
		T = Float{0.0f};
		return 1;
	}
	else
	{
		Float dnp = max(sqrtf(arg), 1e-7f);
		Float nK = (neta * costheta) - dnp;
		T = -(neta * V) + (nK * Nn);
	}

	Float costhetaI{ costheta };
	Float sin2thetaI{ 1.0f - costhetaI * costhetaI };
	Float sin2thetaT{ sin2thetaI / (neta * neta) };
	Float cos2thetaT = sqrtf(1.0f - sin2thetaT);
	Float rp{ (costhetaI - neta * cos2thetaT) / (costhetaI + neta * cos2thetaT) };
	Float rs{ (neta * costhetaI - cos2thetaT) / (neta * costhetaI + cos2thetaT) };

	return 0.5f * (rp * rp + rs * rs);
}



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  SMITH SHADOWING TERM

DEVICE_INL Float lambda(Float absTanTheta, Float alpha)
{
	Float absTanThetaH{ absTanTheta };
	if (isinf(absTanThetaH))
		return 0.0f;
	Float alpha2Tan2Theta{ alpha * absTanThetaH };
	alpha2Tan2Theta *= alpha2Tan2Theta;
	return (-1.0f + sqrtf(1.0f + alpha2Tan2Theta)) / 2.0f;
}

DEVICE_INL Float smithG(Float absTanTheta, Float alpha)
{
	return 1.0f / (1.0f + lambda(absTanTheta, alpha));
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  GENERALIZED TROWBRIDGE-REITZ DISTRIBUTION

DEVICE_INL Float roughnessToAlpha(Float roughness)
{
	return max(MIN_ALPHA, roughness * roughness);
}

DEVICE_INL Float gtr1(Float cosTheta, Float alpha)
{
	if (alpha >= 1.0f) return 1.0f / PI;
	Float alpha2{ alpha * alpha };
	Float t{ 1.0f + (alpha2 - 1.0f) * cosTheta * cosTheta };
	return (alpha2 - 1.0f) / (PI * logf(alpha2) * t);
}

DEVICE_INL Float gtr2(Float cosTheta, Float alpha)
{
	Float alpha2{ alpha * alpha };
	Float t{ 1.0f + (alpha2 - 1.0f) * cosTheta * cosTheta };
	return alpha2 / (PI * t * t);
}

DEVICE Float3 sampleGtr2(Float3 const& V, Float alpha, Float2 u)
{
	// Disney Keynotes eq. (2) and (9) 
	Float alpha2{ alpha * alpha };
	Float phi{ (2 * PI) * u[0] };
	Float cosTheta{ sqrtf((1.0f - u[1]) / (1.0f + (alpha2 - 1.0f) * u[1])) };
	Float sinTheta{ sqrtf(max(0.0f, 1.0f - cosTheta * cosTheta)) };

	Float3 H{ toSphereCoordinates(sinTheta, cosTheta, phi) };
	if (!sameHemisphere(V, H)) H = -H;

	return H;
}

DEVICE Float3 sampleGtr2VNDF(Float3 const& V, Float alpha, Float2 u)
{
	Float3 Vh = normalize(Float3(alpha * V.x, alpha * V.y, V.z));

	float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	Float3 T1 = lensq > 0 ? Float3(-Vh.y, Vh.x, 0) * (1.0f / sqrtf(lensq)) : Float3(1, 0, 0);
	Float3 T2 = cross(Vh, T1);

	float r = sqrtf(u.x);
	float phi = 2.0 * PI * u.y;
	float t1 = r * cos(phi);
	float t2 = r * sin(phi);
	float s = 0.5 * (1.0 + Vh.z);
	t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

	Float3 Nh = t1 * T1 + t2 * T2 + sqrtf(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

	return normalize(Float3(alpha * Nh.x, alpha * Nh.y, max(0.0f, Nh.z)));
}

DEVICE Float pdfGtr2(Float3 const& V, Float3 const& H, Float alpha)
{
	Float Dr{ gtr2(cosTheta(H), alpha)};
	Float Gr{ smithG(abs(tanTheta(V)), alpha)};
	return Dr * Gr * absCosTheta(H) / (4.0f * absCosTheta(V));
}


#endif // !DISNEY_BRDF_UITLS_HPP
