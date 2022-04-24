#ifndef DIELECTRIC_HPP
#define DIELECTRIC_HPP

#include "../Globals.hpp"
#include "../Sampling.hpp"

namespace Dielectric
{
	DEVICE Float3 f(ShadingData& sd, Float3 const& V, Float3 const& L)
	{
		return { 0.0f };

	}

	DEVICE Float3 sampleF(ShadingData& sd, Float3 const& V, Float3& L, Float& pdf)
	{
		return { 0.0f };
	}

	DEVICE void pdf(Float3 const& V, Float3 const& L, Float& pdf)
	{
		pdf = 0.0f;
	}
}

#endif // !DIELECTRIC_HPP
