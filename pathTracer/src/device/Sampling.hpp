#ifndef BA_SAMPLING_HPP
#define BA_SAMPLING_HPP

#include "Types.hpp"
#include <owl/common/math/vec.h>

namespace ba
{
	inline __both__ void makeOrthogonalNormals(owl::vec3f const& N, owl::vec3f& T, owl::vec3f& B)
	{
		if (N.x != N.y || N.x != N.z)
			T = owl::vec3f(N.z - N.y, N.x - N.z, N.y - N.x);	// ( 1, 1, 1) x N
		else
			T = owl::vec3f(N.z - N.y, N.x + N.z, -N.y - N.x);	// (-1, 1, 1) x N

		T = normalize(T);
		B = cross(N, T);
	}

	inline __both__ void uniformSampleDisk(float& x, float& y)
	{
		float phi{ M_2PI_F * y };
		float r{ owl::sqrt(x) };

		x = r * owl::cos(phi);
		y = r * owl::sin(phi);
	}

	inline __both__ void concentricSampleDisk(float& x, float& y)
	{
		// re-scale rand to be between [-1,1]
		float dx { 2.0f * x - 1};
		float dy { 2.0f * y - 1};

		// handle degenerated origin
		if (dx == 0 && dy == 0)
		{
			x = 0;
			y = 0;
			return;
		}

		// handle mapping unit squre to unit disk
		float phi, r;
		if (std::abs(dx) > std::abs(dy))
		{
			r = dx;
			phi = M_PI_OVER_4_F * (dy / dx);
		}
		else
		{
			r = dy;
			phi = M_PI_OVER_2_F - M_PI_OVER_4_F * (dx / dy);
		}

		x = r * std::cos(phi);
		y = r * std::sin(phi);
	}

	inline __both__	void cosineSampleHemisphere(owl::vec3f const &N, float randu, float randv, owl::vec3f &wi, float &pdf)
	{
		// 1. sample unit circle and save position into randu, randv
		concentricSampleDisk(randu, randv);

		// 2. calculate cosTheta => 1 = randu^2 + randv^2 => cos = 1 - (randu^2 + randv^2)
		float cosTheta{ owl::sqrt(owl::max(0.0f, 1.0f - randu * randu - randv * randv)) };

		// 3. create orth. basis to find wi
		owl::vec3f T{};
		owl::vec3f B{};
		makeOrthogonalNormals(N, T, B);

		// 4. set omega_in and pdf of cos_hemisphere
		wi = randu * T + randv * B + cosTheta * N;
		pdf = cosTheta * M_INV_PI_F;
	}
} // namespace ba

#endif // BA_SAMPLING_HPP
