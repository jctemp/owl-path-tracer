#ifndef MATH_HPP
#define MATH_HPP
#pragma once

#include "device.hpp"

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

inline __device__ float sqr(float v) { return v * v; }

#pragma region UTILITY

template<class T>
inline __device__ T mix(T a, T b, T t) { return a + (b - a) * t; }

template<class T>
inline __device__ T saturate(T a);

template<>
inline __device__ float saturate(float a) { return owl::clamp(a, 0.0f, 1.0f); }

template<>
inline __device__ vec3 saturate(vec3 a) {
	return {
		owl::clamp(a.x, 0.0f, 1.0f),
		owl::clamp(a.y, 0.0f, 1.0f),
		owl::clamp(a.z, 0.0f, 1.0f)
	};
}

template<class T>
inline __device__ T pow2(T value) { return value * value; }

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "SHADING SPACE FUNCTIONS"

inline __device__ float cosTheta(vec3 const& w)
{
	return w.z;
}


inline __device__ float cos2Theta(vec3 const& w)
{
	return w.z * w.z;
}


inline __device__ float absCosTheta(vec3 const& w)
{
	return abs(w.z);
}


inline __device__ float sin2Theta(vec3 const& w)
{
	return fmaxf(0.0f, 1.0f - cos2Theta(w));
}


inline __device__ float sinTheta(vec3 const& w)
{
	return sqrtf(sin2Theta(w));
}


inline __device__ float tanTheta(vec3 const& w)
{
	return sinTheta(w) / cosTheta(w);
}


inline __device__ float tan2Theta(vec3 const& w)
{
	return sin2Theta(w) / cos2Theta(w);
}


inline __device__ float cosPhi(vec3 const& w)
{
	float theta{ sinTheta(w) };
	return (theta == 0) ? 1.0f : owl::clamp(w.x / theta, -1.0f, 1.0f);
}


inline __device__ float sinPhi(vec3 const& w)
{
	float theta{ sinTheta(w) };
	return (theta == 0) ? 1.0f : owl::clamp(w.y / theta, -1.0f, 1.0f);
}


inline __device__ float cos2Phi(vec3 const& w)
{
	return cosPhi(w) * cosPhi(w);
}


inline __device__ float sin2Phi(vec3 const& w)
{
	return sinPhi(w) * sinPhi(w);
}


inline __device__ float cosDPhi(vec3 const& wa, vec3 const& wb) {
	return owl::clamp((wa.x * wb.x + wa.y * wb.y) /
		sqrtf((wa.x * wa.x + wa.y * wa.y) *
			(wb.x * wb.x + wb.y * wb.y)), -1.0f, 1.0f);
}


inline __device__ vec3 toSphereCoordinates(float theta, float phi)
{
	float x = sinf(theta) * cosf(phi);
	float y = sinf(theta) * sinf(phi);;
	float z = cosf(theta);
	return vec3{ x, y, z };
}


inline __device__ vec3 toSphereCoordinates(float sinTheta, float cosTheta, float phi)
{
	float x = sinTheta * cosf(phi);
	float y = sinTheta * sinf(phi);;
	float z = cosTheta;
	return vec3{ x, y, z };
}


inline __device__ vec3 reflect(vec3 const& V, vec3 const& N)
{
	return (2.0f * dot(V, N)) * N - V;
}


inline __device__ vec3 refract(vec3 const& V, vec3 const& N, float eta)
{
	float cosThetaI{ dot(V, N) };
	float sin2ThetaI{ fmax(0.0f, 1.0f - cosThetaI * cosThetaI) };
	float sin2ThetaT{ eta * eta * sin2ThetaI };

	if (sin2ThetaT >= 1.0f) return { 0.0f };
	float cosThetaT{ sqrtf(1.0f - sin2ThetaT) };
	return eta * -V + (eta * cosThetaI - cosThetaT) * N;
}

inline __device__ bool refract(vec3 const& V, vec3 const& N, float eta, vec3& T)
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
	float sin2ThetaI{ fmax(0.0f, 1.0f - cosThetaI * cosThetaI) };
	float sin2ThetaT{ eta * eta * sin2ThetaI };

	if (sin2ThetaT >= 1.0f) return false;

	float cosThetaT{ sqrtf(1.0f - sin2ThetaT) };
	T = eta * -V + (eta * cosThetaI - cosThetaT) * N;
	return true;
}



inline __device__ bool sameHemisphere(vec3 const& V, vec3 const& L, vec3 const& N)
{
	return dot(V, N) * dot(L, N) > 0.0f;
}


inline __device__ bool sameHemisphere(vec3 const& V, vec3 const& L)
{
	return V.z * L.z > 0.0f;
}

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "ORTH. NORMAL BASIS"

inline __device__ void onb(vec3 const& N, vec3& T, vec3& B)
{
	if (N.x != N.y || N.x != N.z)
		T = vec3(N.z - N.y, N.x - N.z, N.y - N.x);	// ( 1, 1, 1) x N
	else
		T = vec3(N.z - N.y, N.x + N.z, -N.y - N.x);	// (-1, 1, 1) x N

	T = normalize(T);
	B = cross(N, T);
}


// move vector V to local space where N is (0,0,1)
inline __device__ void toLocal(vec3 const& T, vec3 const& B, vec3 const& N, vec3& V)
{
	V = normalize(vec3{ dot(V, T), dot(V, B), dot(V, N) });
}


// move V from local to the global space
inline __device__ void toWorld(vec3 const& T, vec3 const& B, vec3 const& N, vec3& V)
{
	V = normalize(vec3{ V.x * T + V.y * B + V.z * N });
}

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


#endif // !MATH_HPP
