﻿#ifndef MATH_HPP
#define MATH_HPP
#pragma once

#include "DeviceGlobals.hpp"

#define PI           3.14159265358979323f // pi
#define TWO_PI       6.28318530717958648f // 2pi
#define PI_OVER_TWO  1.57079632679489661f // pi / 2
#define PI_OVER_FOUR 0.78539816339744830f // pi / 4
#define INV_PI       0.31830988618379067f // 1 / pi
#define INV_TWO_PI   0.15915494309189533f // 1 / (2pi)
#define INV_FOUR_PI  0.07957747154594766f // 1 / (4pi)
#define EPSILON      1E-5f
#define T_MIN        1E-3f
#define T_MAX        1E10f
#define MIN_ROUGHNESS 0.04f
#define MIN_ALPHA  MIN_ROUGHNESS * MIN_ROUGHNESS

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region UTILITY

template<class T>
DEVICE_INL T mix(T a, T b, Float t) { return a + (b - a) * t; }

template<class T>
DEVICE_INL T saturate(T a);

template<>
DEVICE_INL Float saturate(Float a) { return owl::clamp(a, 0.0f, 1.0f); }

template<>
DEVICE_INL Float3 saturate(Float3 a) {
	return {
		owl::clamp(a.x, 0.0f, 1.0f),
		owl::clamp(a.y, 0.0f, 1.0f),
		owl::clamp(a.z, 0.0f, 1.0f)
	};
}

template<class T>
DEVICE_INL T pow2(T value) { return value * value; }

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "SHADING SPACE FUNCTIONS"

DEVICE_INL Float cosTheta(Float3 const& w)
{
	return w.z;
}


DEVICE_INL Float cos2Theta(Float3 const& w)
{
	return w.z * w.z;
}


DEVICE_INL Float absCosTheta(Float3 const& w)
{
	return abs(w.z);
}


DEVICE_INL Float sin2Theta(Float3 const& w)
{
	return fmaxf(0.0f, 1.0f - cos2Theta(w));
}


DEVICE_INL Float sinTheta(Float3 const& w)
{
	return sqrtf(sin2Theta(w));
}


DEVICE_INL Float tanTheta(Float3 const& w)
{
	return sinTheta(w) / cosTheta(w);
}


DEVICE_INL Float tan2Theta(Float3 const& w)
{
	return sin2Theta(w) / cos2Theta(w);
}


DEVICE_INL Float cosPhi(Float3 const& w)
{
	Float theta{ sinTheta(w) };
	return (theta == 0) ? 1.0f : owl::clamp(w.x / theta, -1.0f, 1.0f);
}


DEVICE_INL Float sinPhi(Float3 const& w)
{
	Float theta{ sinTheta(w) };
	return (theta == 0) ? 1.0f : owl::clamp(w.y / theta, -1.0f, 1.0f);
}


DEVICE_INL Float cos2Phi(Float3 const& w)
{
	return cosPhi(w) * cosPhi(w);
}


DEVICE_INL Float sin2Phi(Float3 const& w)
{
	return sinPhi(w) * sinPhi(w);
}


DEVICE_INL Float cosDPhi(Float3 const& wa, Float3 const& wb) {
	return owl::clamp((wa.x * wb.x + wa.y * wb.y) /
		sqrtf((wa.x * wa.x + wa.y * wa.y) *
			(wb.x * wb.x + wb.y * wb.y)), -1.0f, 1.0f);
}


DEVICE_INL Float3 toSphereCoordinates(Float theta, Float phi)
{
	Float x = sinf(theta) * cosf(phi);
	Float y = sinf(theta) * sinf(phi);;
	Float z = cosf(theta);
	return Float3{ x, y, z };
}


DEVICE_INL Float3 toSphereCoordinates(Float sinTheta, Float cosTheta, Float phi)
{
	Float x = sinTheta * cosf(phi);
	Float y = sinTheta * sinf(phi);;
	Float z = cosTheta;
	return Float3{ x, y, z };
}


DEVICE_INL Float3 reflect(Float3 const& V, Float3 const& N)
{
	return (2.0f * dot(V, N)) * N - V;
}


DEVICE_INL Float3 reflect(Float3 const& v)
{
	return Float3(-v.x, v.y, -v.z);
}


DEVICE_INL bool sameHemisphere(Float3 const& V, Float3 const& L, Float3 const& N)
{
	return dot(V, N) * dot(L, N) > 0.0f;
}


DEVICE_INL bool sameHemisphere(Float3 const& V, Float3 const& L)
{
	return V.z * L.z > 0.0f;
}

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "ORTH. NORMAL BASIS"

DEVICE_INL void onb(Float3 const& N, Float3& T, Float3& B)
{
	if (N.x != N.y || N.x != N.z)
		T = Float3(N.z - N.y, N.x - N.z, N.y - N.x);	// ( 1, 1, 1) x N
	else
		T = Float3(N.z - N.y, N.x + N.z, -N.y - N.x);	// (-1, 1, 1) x N

	T = normalize(T);
	B = cross(N, T);
}


// move vector V to local space where N is (0,0,1)
DEVICE_INL void toLocal(Float3 const& T, Float3 const& B, Float3 const& N, Float3& V)
{
	V = normalize(Float3{ dot(V, T), dot(V, B), dot(V, N) });
}


// move V from local to the global space
DEVICE_INL void toWorld(Float3 const& T, Float3 const& B, Float3 const& N, Float3& V)
{
	V = normalize(Float3{ V.x * T + V.y * B + V.z * N });
}

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


#endif // !MATH_HPP
