#ifndef PATH_TRACER_MATH_HPP
#define PATH_TRACER_MATH_HPP

#include "device/device.hpp"

inline __both__ float lerp(float a, float b, float t) { return a + (b - a) * t; }
inline __both__ vec3 lerp(vec3 a, vec3 b, vec3 t) { return a + (b - a) * t; }
inline __both__ vec3 lerp(vec3 a, vec3 b, float t) { return a + (b - a) * t; }

inline __both__ float o_saturate(float a) { return owl::clamp(a, 0.0f, 1.0f); }
inline __both__ vec3 o_saturate(vec3 a) { return owl::clamp(a, vec3{0.0f}, vec3{1.0f}); }

inline __both__ float sqr(float v) { return v * v; }

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

inline __both__ float cos_theta(vec3 const& w){	return w.z; }

inline __both__ float sin_theta(vec3 const& w){	return owl::sqrt(owl::max(0.0f, 1.0f - sqr(cos_theta(w)))); }

inline __both__ float tan_theta(vec3 const& w){	return sin_theta(w) / cos_theta(w); }

inline __both__ float cos_phi(vec3 const& w)
{
	float theta{ sin_theta(w) };
	return (theta == 0) ? 1.0f : owl::clamp(w.x / theta, -1.0f, 1.0f);
}

inline __both__ float sin_phi(vec3 const& w)
{
	float theta{ sin_theta(w) };
	return (theta == 0) ? 1.0f : owl::clamp(w.y / theta, -1.0f, 1.0f);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

inline __both__ vec3 toSphereCoordinates(float theta, float phi)
{
	float x = sinf(theta) * cosf(phi);
	float y = sinf(theta) * sinf(phi);;
	float z = cosf(theta);
	return vec3{ x, y, z };
}


inline __both__ vec3 toSphereCoordinates(float sinTheta, float cosTheta, float phi)
{
	float x = sinTheta * cosf(phi);
	float y = sinTheta * sinf(phi);;
	float z = cosTheta;
	return vec3{ x, y, z };
}


inline __both__ vec3 reflect(vec3 const& V, vec3 const& N)
{
	return (2.0f * dot(V, N)) * N - V;
}


inline __both__ vec3 refract(vec3 const& V, vec3 const& N, float eta)
{
	float cosThetaI{ dot(V, N) };
	float sin2ThetaI{ fmax(0.0f, 1.0f - cosThetaI * cosThetaI) };
	float sin2ThetaT{ eta * eta * sin2ThetaI };

	if (sin2ThetaT >= 1.0f) return { 0.0f };
	float cosThetaT{ sqrtf(1.0f - sin2ThetaT) };
	return eta * -V + (eta * cosThetaI - cosThetaT) * N;
}

inline __both__ bool refract(vec3 const& V, vec3 const& N, float eta, vec3& T)
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



inline __both__ bool sameHemisphere(vec3 const& V, vec3 const& L, vec3 const& N)
{
	return dot(V, N) * dot(L, N) > 0.0f;
}


inline __both__ bool sameHemisphere(vec3 const& V, vec3 const& L)
{
	return V.z * L.z > 0.0f;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


inline __both__ void onb(vec3 const& N, vec3& T, vec3& B)
{
	if (N.x != N.y || N.x != N.z)
		T = vec3(N.z - N.y, N.x - N.z, N.y - N.x);	// ( 1, 1, 1) x N
	else
		T = vec3(N.z - N.y, N.x + N.z, -N.y - N.x);	// (-1, 1, 1) x N

	T = normalize(T);
	B = cross(N, T);
}


// move vector V to local space where N is (0,0,1)
inline __both__ void toLocal(vec3 const& T, vec3 const& B, vec3 const& N, vec3& V)
{
	V = normalize(vec3{ dot(V, T), dot(V, B), dot(V, N) });
}


// move V from local to the global space
inline __both__ void toWorld(vec3 const& T, vec3 const& B, vec3 const& N, vec3& V)
{
	V = normalize(vec3{ V.x * T + V.y * B + V.z * N });
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


#endif // !MATH_HPP
