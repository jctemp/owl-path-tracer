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
//           DISNEY PRINCIPLED DIFFUSE

template<>
DEVICE void f<Material::BRDF_DIFFUSE>(MaterialStruct& ms, Float3 const& V, Float3 const& L,
	Float3& brdf)
{
	Float3 H = normalize(V + L);
	Float NdotV = absCosTheta(V);
	Float NdotL = absCosTheta(L);
	Float LdotH = dot(L, H);

	Float fl = schlickFresnel(NdotL);
	Float fv = schlickFresnel(NdotV);

	/* DIFFUSE REFLECTANCE LOBE */

	Float fd90 = 0.5f + 1.0f * ms.roughness * LdotH * LdotH;
	Float fdV = mix(1.f, fd90, fv);
	Float fdL = mix(1.f, fd90, fl);
	Float3 fd = ms.baseColor * INV_PI * fdV * fdL;

	/* SUBSURFACE SCATTERING LOBE */

	Float fss90{ ms.roughness * LdotH * LdotH };
	Float fssL = mix(1.0f, fss90, fl);
	Float fssV = mix(1.0f, fss90, fv);
	// 1.0f / (NdotV + NdotL) => volumetric absorption scattering media below surface
	Float3 fs = 1.25f * ms.subsurfaceColor * INV_PI * fssL * fssV * ( 1.0f / (NdotV + NdotL) - 0.5f) + 0.5f;

	// one could change the baseColor to linear space with pow(color, 2.2f)
	brdf = Float3{ (1 - ms.subsurface) * fd + ms.subsurface * fs};
}

template<>
DEVICE void sampleF<Material::BRDF_DIFFUSE>(MaterialStruct& ms, Float3 const& V, Float2 u, Float3& L,
	Float3& brdf, Float& pdf)
{
	sampleCosineHemisphere({ u.x ,u.v }, L);
	pdfCosineHemisphere(V, L, pdf);
	f<Material::BRDF_DIFFUSE>(ms, V, L, brdf);
}

template<>
DEVICE void pdf<Material::BRDF_DIFFUSE>(Float3 const& V, Float3 const& L,
	Float& pdf)
{
	if (!sameHemisphere(V, L)) pdf = 0.0f;
	else pdfCosineHemisphere(V, L, pdf);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           DISNEY PRINCIPLED GLOSSY







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
