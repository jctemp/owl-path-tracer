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

inline __both__ vec3 to_sphere_coordinates(float theta, float phi)
{
	float x = owl::sin(theta) * owl::cos(phi);
	float y = owl::sin(theta) * owl::sin(phi);;
	float z = owl::cos(theta);
	return vec3{ x, y, z };
}

inline __both__ vec3 to_sphere_coordinates(float sin_theta, float cos_theta, float phi)
{
	float x = sin_theta * owl::cos(phi);
	float y = sin_theta * owl::sin(phi);;
	float z = cos_theta;
	return vec3{ x, y, z };
}

inline __both__ vec3 reflect(vec3 const& w, vec3 const& n)
{
	return (2.0f * owl::dot(w, n)) * n - w;
}

inline __both__ bool same_hemisphere(vec3 const& w_o, vec3 const& w_i)
{
	return w_o.z * w_i.z > 0.0f;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

inline __both__ void onb(vec3 const& n, vec3& t, vec3& b)
{
    // ( 1, 1, 1) x N
	if (n.x != n.y || n.x != n.z) t = vec3(n.z - n.y, n.x - n.z, n.y - n.x);
    // (-1, 1, 1) x N
	else t = vec3(n.z - n.y, n.x + n.z, -n.y - n.x);

	t = owl::normalize(t);
    b = owl::cross(n, t);
}

// move vector V to local space where N is (0,0,1)
inline __both__ void to_local(vec3 const& t, vec3 const& b, vec3 const& n, vec3& w)
{
    w = owl::normalize(vec3{owl::dot(w, t), owl::dot(w, b), owl::dot(w, n) });
}

// move V from local to the global space
inline __both__ void to_world(vec3 const& t, vec3 const& b, vec3 const& n, vec3& w)
{
    w = owl::normalize(vec3{w.x * t + w.y * b + w.z * n });
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


#endif // !MATH_HPP
