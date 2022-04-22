#ifndef BSDF_UTIL_HPP
#define BSDF_UTIL_HPP

#include "Globals.hpp"


DEVICE void makeOrthogonals(Float3 const& N, Float3& T, Float3& B)
{
	if (N.x != N.y || N.x != N.z)
		T = Float3(N.z - N.y, N.x - N.z, N.y - N.x);	// ( 1, 1, 1) x N
	else
		T = Float3(N.z - N.y, N.x + N.z, -N.y - N.x);	// (-1, 1, 1) x N

	T = normalize(T);
	B = cross(N, T);
}

template<class T>
DEVICE T mix(T a, T b, Float t)
{
	return a + t * (b - a);
}


template<class T>
DEVICE T saturate(T a);


template<>
DEVICE Float saturate(Float a)
{
	return owl::clamp(a, 0.0f, 1.0f);
}


template<>
DEVICE Float3 saturate(Float3 a)
{
	return {
		owl::clamp(a.x, 0.0f, 1.0f),
		owl::clamp(a.y, 0.0f, 1.0f),
		owl::clamp(a.z, 0.0f, 1.0f)
	};
}


DEVICE Float schlickWeight(Float u)
{
	return pow(saturate<Float>(1.0f - u), 5.0f);
}


#endif // BSDF_UTIL_HPP 
