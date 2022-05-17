#ifndef DISNEY_BRDF_UITLS_HPP
#define DISNEY_BRDF_UITLS_HPP
#pragma once

#include "../Sampling.hpp"

// SOURCES
// https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf
// https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.2297&rep=rep1&type=pdf
// https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
// https://jcgt.org/published/0007/04/01/

PT_DEVICE_INLINE float luminance(vec3 color)
{
	return 0.212671f * color.x + 0.715160f * color.y + 0.072169f * color.z;
}

PT_DEVICE_INLINE vec3 calculateTint(vec3 baseColor)
// diseny uses in the BRDF explorer the luminance for the calculation
{
	float lum{ luminance(baseColor) };
	return (lum > 0.0f) ? baseColor * (1.0f / lum) : vec3{ 1.0f };
}

PT_DEVICE_INLINE float calculateEta(vec3 V, float ior)
{
	if (cosTheta(V) > 0.0f)
		return 1.0f / ior;
	return ior;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  FRESNEL TERM

PT_DEVICE_INLINE float schlickFresnel(float costheta, float ior)
{
	float r0{ (1.0f - ior) / (1.0f + ior) };
	r0 = r0 * r0;
	float m{ owl::clamp(1.0f - costheta, 0.0f, 1.0f) };
	float m2{ m * m };
	return r0 + (1.0f - r0) * (m2 * m2 * m);
}

PT_DEVICE_INLINE float dielectricFresnel(float costheta, float ior)
{
	float costhetaI{ costheta };

	if (costhetaI < 0.0f)
	{
		costhetaI = -costhetaI;
		ior = 1.0f / ior;
	}

	float sin2thetaI{ 1.0f - costhetaI * costhetaI };
	float sin2thetaT{ sin2thetaI / (ior * ior) };

	if (sin2thetaT > 1.0f)
	{
		return 1.0f;
	}

	float cos2thetaT = sqrtf(1.0f - sin2thetaT);
	float rp{ (costhetaI - ior * cos2thetaT) / (costhetaI + ior * cos2thetaT) };
	float rs{ (ior * costhetaI - cos2thetaT) / (ior * costhetaI + cos2thetaT) };

	return 0.5f * (rp * rp + rs * rs);
}

PT_DEVICE_INLINE float dielectricFresnel(float ior, vec3 V, vec3 N, vec3 & R, vec3& T, bool& inside)
{
	float costheta{ cosTheta(V) }, neta{};
	vec3 Nn;

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

	float arg = 1 - (neta * neta * (1 - (costheta * costheta)));
	if (arg < 0) 
	{
		T = float{0.0f};
		return 1;
	}
	else
	{
		float dnp = max(sqrtf(arg), 1e-7f);
		float nK = (neta * costheta) - dnp;
		T = -(neta * V) + (nK * Nn);
	}

	float costhetaI{ costheta };
	float sin2thetaI{ 1.0f - costhetaI * costhetaI };
	float sin2thetaT{ sin2thetaI / (neta * neta) };
	float cos2thetaT = sqrtf(1.0f - sin2thetaT);
	float rp{ (costhetaI - neta * cos2thetaT) / (costhetaI + neta * cos2thetaT) };
	float rs{ (neta * costhetaI - cos2thetaT) / (neta * costhetaI + cos2thetaT) };

	return 0.5f * (rp * rp + rs * rs);
}



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  SMITH SHADOWING TERM

PT_DEVICE_INLINE float lambda(float absTanTheta, float alpha)
{
	float absTanThetaH{ absTanTheta };
	if (isinf(absTanThetaH))
		return 0.0f;
	float alpha2Tan2Theta{ alpha * absTanThetaH };
	alpha2Tan2Theta *= alpha2Tan2Theta;
	return (-1.0f + sqrtf(1.0f + alpha2Tan2Theta)) / 2.0f;
}

PT_DEVICE_INLINE float smithG(float absTanTheta, float alpha)
{
	return 1.0f / (1.0f + lambda(absTanTheta, alpha));
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  GENERALIZED TROWBRIDGE-REITZ DISTRIBUTION

PT_DEVICE_INLINE float roughnessToAlpha(float roughness)
{
	return max(MIN_ALPHA, roughness * roughness);
}

PT_DEVICE_INLINE float gtr1(float cosTheta, float alpha)
{
	if (alpha >= 1.0f) return 1.0f / PI;
	float alpha2{ alpha * alpha };
	float t{ 1.0f + (alpha2 - 1.0f) * cosTheta * cosTheta };
	return (alpha2 - 1.0f) / (PI * logf(alpha2) * t);
}

PT_DEVICE_INLINE float gtr2(float cosTheta, float alpha)
{
	float alpha2{ alpha * alpha };
	float t{ 1.0f + (alpha2 - 1.0f) * cosTheta * cosTheta };
	return alpha2 / (PI * t * t);
}

PT_DEVICE_INLINE vec3 sampleGtr2(vec3 const& V, float alpha, vec2 u)
{
	// Disney Keynotes eq. (2) and (9) 
	float alpha2{ alpha * alpha };
	float phi{ (2 * PI) * u[0] };
	float cosTheta{ sqrtf((1.0f - u[1]) / (1.0f + (alpha2 - 1.0f) * u[1])) };
	float sinTheta{ sqrtf(max(0.0f, 1.0f - cosTheta * cosTheta)) };

	vec3 H{ toSphereCoordinates(sinTheta, cosTheta, phi) };
	if (!sameHemisphere(V, H)) H = -H;

	return H;
}

PT_DEVICE_INLINE vec3 sampleGtr2VNDF(vec3 const& V, float alpha, vec2 u)
{
	vec3 Vh = normalize(vec3(alpha * V.x, alpha * V.y, V.z));

	float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	vec3 T1 = lensq > 0 ? vec3(-Vh.y, Vh.x, 0) * (1.0f / sqrtf(lensq)) : vec3(1, 0, 0);
	vec3 T2 = cross(Vh, T1);

	float r = sqrtf(u.x);
	float phi = 2.0 * PI * u.y;
	float t1 = r * cos(phi);
	float t2 = r * sin(phi);
	float s = 0.5 * (1.0 + Vh.z);
	t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

	vec3 Nh = t1 * T1 + t2 * T2 + sqrtf(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

	return normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0f, Nh.z)));
}

PT_DEVICE_INLINE float pdfGtr2(vec3 const& V, vec3 const& H, float alpha)
{
	float Dr{ gtr2(cosTheta(H), alpha)};
	float Gr{ smithG(abs(tanTheta(V)), alpha)};
	return Dr * Gr * absCosTheta(H) / (4.0f * absCosTheta(V));
}


#endif // !DISNEY_BRDF_UITLS_HPP
