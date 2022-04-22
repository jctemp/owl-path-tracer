#ifndef SAMPLING_HPP
#define SAMPLING_HPP

#include "Globals.hpp"
#include "BsdfUtil.hpp"


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


DEVICE void sampleCosineHemisphere(Float3 const& N,
	Float2& rand, Float3& L, Float& pdf)
{
	// 1. sample unit circle and save position into randu, randv
	sampleConcentricDisk(rand);

	// 2. calculate cosTheta => 1 = randu^2 + randv^2 => cos = 1 - (randu^2 + randv^2)
	Float cosTheta{ owl::sqrt(owl::max(0.0f, 1.0f - rand.x * rand.x - rand.y * rand.y)) };

	// 3. move to world space with 
	// V.x * T + V.y * B + V.z * N;
	Float3 T{}, B{};
	makeOrthogonals(N, T, B);
	
	L = rand.x * T + rand.y * B + cosTheta * N;
	pdf = cosTheta * INV_PI;
}


DEVICE void sampleUniformHemisphere(Float3 const& N,
	Float2& rand, Float3& L, Float& pdf)
{
	Float z{ rand.x };
	Float r{ sqrtf(max(0.0f, 1.0f - z * z)) };
	Float phi = TWO_PI * rand.y;

	Float x = r * owl::cos(phi);
	Float y = r * owl::sin(phi);

	Float3 T{}, B{};
	makeOrthogonals(N, T, B);

	L = x * T + y * B + z * N;
	pdf = 0.5f * INV_PI;
}


DEVICE Float3 sampleUniformSphere(Float2 rand)
{
	Float z { 1.0f - 2.0f * rand.x };
	Float r { sqrtf(fmaxf(0.0f, 1.0f - z * z)) };
	Float phi { TWO_PI * rand.y };
	Float x = r * owl::cos(phi);
	Float y = r * owl::sin(phi);

	return Float3{ x, y, z };
}


#endif // SAMPLING_HPP
