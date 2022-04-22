#ifndef DISNEY_HPP
#define DISNEY_HPP

#include "Globals.hpp"
#include "Sampling.hpp"
#include "BsdfUtil.hpp"
#include "Random.hpp"




DEVICE Float3 diffuseColor(MaterialData const& mat, Float3 const& N, Float3 const& V,
	Float3 const& L, Float3 const& H)
{
	return mat.baseColor;
}

DEVICE Float3 subsurfaceColor(MaterialData const& mat, Float3 const& N, Float3 const& V,
	Float3 const& L, Float3 const& H)
{
	return mat.subsurface;
}


DEVICE void diffuse(MaterialData const& mat, Float3 const& N, Float3 const& V, 
	Float3 const& L, Float3 const& H, Float3& bsdf, Float3& color)
{
	Float NdotL{ fabs(dot(N, L)) };
	Float NdotV{ fabs(dot(N, V)) };
	Float LdotH{ dot(L, H) };

	Float Fd90{ 0.5f + 1.0f * mat.roughness * LdotH * LdotH };
	Float FL{ schlickWeight(NdotL) };
	Float FV{ schlickWeight(NdotV) };

	color = diffuseColor(mat, N, V, L, H);
	bsdf = Float3{ INV_PI * mix(1.0f, Fd90, FL) * mix(1.0f, Fd90, FV) };
}

DEVICE void subsurface(MaterialData const& mat, Float3 const& N, Float3 const& V, 
	Float3 const& L, Float3 const& H, Float3& bsdf, Float3& color) 
{
	Float NdotL{ fabs(dot(N, L)) };
	Float NdotV{ fabs(dot(N, V)) };
	Float LdotH{ dot(L, H) };

	Float FL{ schlickWeight(NdotL) };
	Float FV{ schlickWeight(NdotV) };

	Float Fss90 = LdotH * LdotH * mat.roughness;
	Float Fss = mix(1.0f, Fss90, FL) * mix(1.0f, Fss90, FV);
	Float ss = 1.25 * (Fss * (1. / (NdotL + NdotV) - .5) + .5);

	color = subsurfaceColor(mat, n, w_o, w_i);
	bsdf = Float3{ INV_PI * ss };
}


#endif // DISNEY_HPP