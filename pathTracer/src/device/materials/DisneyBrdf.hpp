#ifndef DISNEY_BRDF_HPP
#define DISNEY_BRDF_HPP
#pragma once

#include "Materials.hpp"
#include "../Sampling.hpp"
#include "DisneyBrdfUtils.hpp"

// REFECRENCES:
// https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// 


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           DISNEY PRINCIPLED BRDF


/* DIFFUSE REFLECTANCE LOBE */
// https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf
DEVICE void disneyDiffuse(MaterialStruct const& mat, Float3 const& N, Float3 const& V,
	Float3 const& L, Float3 const& H, Float3& bsdf, Float3& color)
{
	Float NdotV{ absCosTheta(V) };
	Float NdotL{ absCosTheta(L) };
	Float LdotH{ dot(L, H) };

	Float fl{ schlickFresnel(NdotL) };
	Float fv{ schlickFresnel(NdotV) };

	Float fd90{ 0.5f + 1.0f * mat.roughness * LdotH * LdotH };
	Float fdV{ mix(1.f, fd90, fv) };
	Float fdL{ mix(1.f, fd90, fl) };
	Float3 fd{ INV_PI * fdV * fdL };

	bsdf = fd;
	color = mat.baseColor;
}


/* SUBSURFACE SCATTERING LOBE */
DEVICE void disneySubsurface(MaterialStruct const& mat, Float3 const& N, Float3 const& V,
	Float3 const& L, Float3 const& H, Float3& bsdf, Float3& color)
{
	Float NdotV{ absCosTheta(V) };
	Float NdotL{ absCosTheta(L) };
	Float LdotH{ dot(L, H) };

	Float fl{ schlickFresnel(NdotL) };
	Float fv{ schlickFresnel(NdotV) };

	Float fss90{ mat.roughness * LdotH * LdotH };
	Float fssL{ mix(1.0f, fss90, fl) };
	Float fssV{ mix(1.0f, fss90, fv) };
	Float3 fs{ 1.25f * INV_PI * fssL * fssV * (1.0f / (NdotV + NdotL) - 0.5f) + 0.5f };
	// 1.0f / (NdotV + NdotL) => volumetric absorption scattering media below surface

	bsdf = fs;
	color = mat.subsurfaceColor;
}


template<>
DEVICE void f<Material::BRDF_DIFFUSE>(MaterialStruct const& mat, Float3 const& Ng, Float3 const& N,
	Float3 const& T, Float3 const& B, Float3 const& V, Float3 const& L, Float3 const& H,
	Float3& bsdf)
{
	Float3 diffuseBsdf{}, diffuseColor{};
	disneyDiffuse(mat, N, V, L, H, diffuseBsdf, diffuseColor);
	
	Float3 subsurfaceBsdf{}, subsurfaceColor{};
	disneySubsurface(mat, N, V, L, H, subsurfaceBsdf, subsurfaceColor);

	bsdf = mix(diffuseBsdf * diffuseColor, subsurfaceBsdf * subsurfaceColor, mat.subsurface);
}

template<>
DEVICE void sampleF<Material::BRDF_DIFFUSE>(MaterialStruct const& mat, Random& random, 
	Float3 const& Ng, Float3 const& N, Float3 const& T, Float3 const& B, Float3 const& V, 
	Float3& L, Float& pdf, Float3& bsdf)
{
	sampleCosineHemisphere({ random() ,random() }, L);
	pdfCosineHemisphere(V, L, pdf);
	Float3 H{ normalize(L + V) };
	f<Material::BRDF_DIFFUSE>(mat, Ng, N, T, B, V, L, H, bsdf);
}

template<>
DEVICE void pdf<Material::BRDF_DIFFUSE>(Float3 const& V, Float3 const& L,
	Float& pdf)
{
	if (!sameHemisphere(V, L)) pdf = 0.0f;
	else pdfCosineHemisphere(V, L, pdf);
}





/*
* MICROFACET MODELS
*
* f(l, v) = \text{diffuse} + \frac{D(\theta_h)F(\theta_d)G(\theta_l, \theta_v)}{4\cos\theta_l\cos\theta_v}
* - D is responsible for specular peak				=> GGX distribution
* - F is reflection coefficient (fresnel)			=> fresnel-schlick approximation
* - G is geometric attenuation or shadowing factor	=> GGX shadowing or visibility term (self-shadowing)
* => using GGX because of short peaks and long tails for specular
*
* - \theta_l and \theta_v angles of incident of in coming and out going vector (l, v)
* - \theta_h angle between half-vector and normal
* - \theta_d differens between l and h or v and h
*
* Microfacet model is energy conserving allows for one parameter roughness! Defines behaviour for
* specular reflections or refractions of an object.
* The NDF is dominant component in the model which drives the roughness appearance most.
*
* 1. Sample random microfacet normal
* 2. reflect exitence radiance along normal to get incident direction (reflect V to get L)
*
* Different Types:
*	- GGX
*	- Beckmann
*	- Blinn
*/






#endif // !DISNEY_BRDF_HPP
