﻿#ifndef MATH_HPP
#define MATH_HPP
#pragma once

#include "device.hpp"

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region UTILITY

template<class T>
PT_DEVICE_INLINE T mix(T a, T b, T t) { return a + (b - a) * t; }

template<class T>
PT_DEVICE_INLINE T saturate(T a);

template<>
PT_DEVICE_INLINE float saturate(float a) { return owl::clamp(a, 0.0f, 1.0f); }

template<>
PT_DEVICE_INLINE Float3 saturate(Float3 a) {
	return {
		owl::clamp(a.x, 0.0f, 1.0f),
		owl::clamp(a.y, 0.0f, 1.0f),
		owl::clamp(a.z, 0.0f, 1.0f)
	};
}

template<class T>
PT_DEVICE_INLINE T pow2(T value) { return value * value; }

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "SHADING SPACE FUNCTIONS"

PT_DEVICE_INLINE float cosTheta(Float3 const& w)
{
	return w.z;
}


PT_DEVICE_INLINE float cos2Theta(Float3 const& w)
{
	return w.z * w.z;
}


PT_DEVICE_INLINE float absCosTheta(Float3 const& w)
{
	return abs(w.z);
}


PT_DEVICE_INLINE float sin2Theta(Float3 const& w)
{
	return fmaxf(0.0f, 1.0f - cos2Theta(w));
}


PT_DEVICE_INLINE float sinTheta(Float3 const& w)
{
	return sqrtf(sin2Theta(w));
}


PT_DEVICE_INLINE float tanTheta(Float3 const& w)
{
	return sinTheta(w) / cosTheta(w);
}


PT_DEVICE_INLINE float tan2Theta(Float3 const& w)
{
	return sin2Theta(w) / cos2Theta(w);
}


PT_DEVICE_INLINE float cosPhi(Float3 const& w)
{
	float theta{ sinTheta(w) };
	return (theta == 0) ? 1.0f : owl::clamp(w.x / theta, -1.0f, 1.0f);
}


PT_DEVICE_INLINE float sinPhi(Float3 const& w)
{
	float theta{ sinTheta(w) };
	return (theta == 0) ? 1.0f : owl::clamp(w.y / theta, -1.0f, 1.0f);
}


PT_DEVICE_INLINE float cos2Phi(Float3 const& w)
{
	return cosPhi(w) * cosPhi(w);
}


PT_DEVICE_INLINE float sin2Phi(Float3 const& w)
{
	return sinPhi(w) * sinPhi(w);
}


PT_DEVICE_INLINE float cosDPhi(Float3 const& wa, Float3 const& wb) {
	return owl::clamp((wa.x * wb.x + wa.y * wb.y) /
		sqrtf((wa.x * wa.x + wa.y * wa.y) *
			(wb.x * wb.x + wb.y * wb.y)), -1.0f, 1.0f);
}


PT_DEVICE_INLINE Float3 toSphereCoordinates(float theta, float phi)
{
	float x = sinf(theta) * cosf(phi);
	float y = sinf(theta) * sinf(phi);;
	float z = cosf(theta);
	return Float3{ x, y, z };
}


PT_DEVICE_INLINE Float3 toSphereCoordinates(float sinTheta, float cosTheta, float phi)
{
	float x = sinTheta * cosf(phi);
	float y = sinTheta * sinf(phi);;
	float z = cosTheta;
	return Float3{ x, y, z };
}


PT_DEVICE_INLINE Float3 reflect(Float3 const& V, Float3 const& N)
{
	return (2.0f * dot(V, N)) * N - V;
}


PT_DEVICE_INLINE Float3 refract(Float3 const& V, Float3 const& N, float eta)
{
	float cosThetaI{ dot(V, N) };
	float sin2ThetaI{ max(0.0f, 1.0f - cosThetaI * cosThetaI) };
	float sin2ThetaT{ eta * eta * sin2ThetaI };

	if (sin2ThetaT >= 1.0f) return { 0.0f };
	float cosThetaT{ sqrtf(1.0f - sin2ThetaT) };
	return eta * -V + (eta * cosThetaI - cosThetaT) * N;
}

PT_DEVICE_INLINE bool refract(Float3 const& V, Float3 const& N, float eta, Float3& T)
{
	T = -V;
	if (eta == 1.0f)return  true;
	if (eta <= 0.0f) return false;
	if (isnan(eta)) return false;
	if (isinf(eta))return  false;
	
	//float costheta = dot(-V, N);
	//Float3 rOutPerp{ eta * (V + costheta * N) };
	//Float3 rOutPara{ sqrtf(max(0.0f, 1.0f - dot(rOutPerp,rOutPerp))) * N };
	//T = rOutPara + rOutPerp;
	//return true;

	float cosThetaI{ dot(-V, N) };
	float sin2ThetaI{ max(0.0f, 1.0f - cosThetaI * cosThetaI) };
	float sin2ThetaT{ eta * eta * sin2ThetaI };

	if (sin2ThetaT >= 1.0f) return false;

	float cosThetaT{ sqrtf(1.0f - sin2ThetaT) };
	T = eta * -V + (eta * cosThetaI - cosThetaT) * N;
	return true;
}



PT_DEVICE_INLINE bool sameHemisphere(Float3 const& V, Float3 const& L, Float3 const& N)
{
	return dot(V, N) * dot(L, N) > 0.0f;
}


PT_DEVICE_INLINE bool sameHemisphere(Float3 const& V, Float3 const& L)
{
	return V.z * L.z > 0.0f;
}

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "ORTH. NORMAL BASIS"

PT_DEVICE_INLINE void onb(Float3 const& N, Float3& T, Float3& B)
{
	if (N.x != N.y || N.x != N.z)
		T = Float3(N.z - N.y, N.x - N.z, N.y - N.x);	// ( 1, 1, 1) x N
	else
		T = Float3(N.z - N.y, N.x + N.z, -N.y - N.x);	// (-1, 1, 1) x N

	T = normalize(T);
	B = cross(N, T);
}


// move vector V to local space where N is (0,0,1)
PT_DEVICE_INLINE void toLocal(Float3 const& T, Float3 const& B, Float3 const& N, Float3& V)
{
	V = normalize(Float3{ dot(V, T), dot(V, B), dot(V, N) });
}


// move V from local to the global space
PT_DEVICE_INLINE void toWorld(Float3 const& T, Float3 const& B, Float3 const& N, Float3& V)
{
	V = normalize(Float3{ V.x * T + V.y * B + V.z * N });
}

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


#endif // !MATH_HPP
