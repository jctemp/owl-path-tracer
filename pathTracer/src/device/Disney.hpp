#ifndef DISNEY_HPP
#define DISNEY_HPP

#include "DeviceGlobals.hpp"
#include "Sampling.hpp"
#include "BsdfUtil.hpp"

// YCbCr model
DEVICE_INL Float luminance(Float3 color)
{
	return 0.212671f * color.x + 0.715160f * color.y + 0.072169f * color.z;
}


DEVICE_INL Float3 pow(Float3 const& base, Float3 const& exponent)
{
	return {
		powf(base.x, exponent.x),
		powf(base.y, exponent.y),
		powf(base.z, exponent.z)
	};
}


DEVICE_INL Float3 linearToGamma(Float3 linearColor) {
	return pow(linearColor, Float3{ 0.4545f });
}


DEVICE_INL Float3 gammaToLinear(Float3 gammaColor) {
	return pow(gammaColor, Float3{ 2.2f });
}


DEVICE_INL Float3 computeHalfVector(Float3 const& V1, Float3 const& V2)
{
	return normalize(V1 + V2);
}


DEVICE_INL bool relativeIOR(Float3 const& V, Float3 const& N, Float const& ior,
	Float& etaO, Float& etaI)
{
	bool entering{ dot(V, N) > 0.f };
	etaI = entering ? 1.f : ior;
	etaO = entering ? ior : 1.f;
	return entering;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           GENERALIZED TROWBRIDGE-REITZ (gamma=1)
//           > Burley notes eq. 4


DEVICE Float GTR1(Float const& cosThetaH, Float const& alpha) {
	if (alpha >= 1.f) {
		return INV_PI;
	}

	Float alpha2{ alpha * alpha };
	return INV_PI * (alpha2 - 1.f) / (log(alpha2) * (1.f + (alpha2 - 1.f)
		* cosThetaH * cosThetaH));
}


DEVICE void pdfGTR1(Float3 const& V, Float3 const& L, Float3 const& H, Float3 const& N,
	Float const& alpha, Float& clearcoat)
{
	if (!sameHemisphere(V, L, N))
	{
		clearcoat = 0.0f;
		return;
	}

	Float cosThetaH{ dot(N, H) };
	Float d{ GTR1(cosThetaH, alpha) };
	clearcoat = d * cosThetaH / (4.f * dot(V, H));
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           GENERALIZED TROWBRIDGE-REITZ (gamma=2)
//           > Burley notes eq. 8


DEVICE Float GTR2(Float const& cosThetaH, Float const& alpha)
{
	Float alpha2{ alpha * alpha };
	return INV_PI * alpha2 / fmaxf(pow2(1.f + (alpha2 - 1.f) * cosThetaH *
		cosThetaH), EPSILON);
}


DEVICE void pdfGTR2(Float3 const& V, Float3 const& L, Float3 const& H, Float3 const& N,
	Float const& alpha, Float& microfacet)
{
	if (!sameHemisphere(V, L, N))
	{
		microfacet = 0.0f;
		return;
	}

	Float cosThetaH{ dot(N, H) };
	Float d{ GTR2(cosThetaH, alpha) };
	microfacet = d * cosThetaH / (4.f * fabs(dot(V, H)));
}


DEVICE void pdfGTR2Transmission(Float3 const& V, Float3 const& L, Float3 const& N,
	Float const& transmissionRoughness, Float const& ior, Float& microfacetTransmission)
{
	if (sameHemisphere(V, L, N))
	{
		microfacetTransmission = 0.0f;
		return;
	}

	Float alpha{ fmaxf(0.001f, transmissionRoughness * transmissionRoughness) };
	Float etaO, etaI;
	bool entering = relativeIOR(V, N, ior, etaO, etaI);

	Float3 Ht{ normalize(-(V * etaI + L * etaO)) };
	Float cosThetaH{ fabs(dot(N, Ht)) };
	Float d{ GTR2(cosThetaH, alpha) };
	microfacetTransmission = d;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//           ANISOTROPIC GENERALIZED TROWBRIDGE-REITZ (gamma=2)
//          > Burley notes eq. 13


DEVICE Float GTR2Aniso(Float const& cosThetaH, Float const& cosThetaT, Float const& cosThetaB,
	Float2 const& alpha)
{
	return INV_PI / fmaxf((alpha.x * alpha.y * pow2(pow2(cosThetaT / alpha.x) +
		pow2(cosThetaB / alpha.y) + cosThetaH * cosThetaH)), EPSILON);
}


DEVICE void pdfGTR2Aniso(Float3 const& V, Float3 const& L, Float3 const& H, Float3 const& N,
	Float3 const& T, Float3 const& B, Float2 const& alphaAniso, Float& microfacet)
{
	if (!sameHemisphere(V, L, N))
	{
		microfacet = 0.0f;
		return;
	}

	Float cos_theta_h = dot(N, H);
	Float d{ GTR2Aniso(cos_theta_h, fabs(dot(H, T)), fabs(dot(H, B)), alphaAniso) };
	microfacet = d * cos_theta_h / (4.f * dot(V, L));
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


DEVICE Float3 disneyDiffuseColor(Material const& mat, Float3 const& N, Float3 const& V,
	Float3 const& L, Float3 const& H)
{
	return mat.baseColor;
}

DEVICE void disneyDiffuse(Material const& mat, Float3 const& N, Float3 const& V, 
	Float3 const& L, Float3 const& H, Float3& bsdf, Float3& color)
{
	Float NdotV{ fabsf(dot(N, V)) };
	Float NdotL{ fabsf(dot(N, L)) };
	Float NdotH{ fabsf(dot(N, H)) };

	Float Fd90{ 0.5f + 1.0f * mat.roughness * NdotH * NdotH };
	Float FV{ schlickWeight(NdotV) };
	Float FL{ schlickWeight(NdotL) };

	color = disneyDiffuseColor(mat, N, V, L, H);
	bsdf = Float3{ INV_PI * mix(1.0f, Fd90, FL) * mix(1.0f, Fd90, FV) };
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


DEVICE Float3 disneySubsurfaceColor(Material const& mat, Float3 const& N, Float3 const& V,
	Float3 const& L, Float3 const& H)
{
	return mat.subsurfaceColor;
}

DEVICE void disneySubsurface(Material const& mat, Float3 const& N, Float3 const& V,
	Float3 const& L, Float3 const& H, Float3& bsdf, Float3& color)
{
	Float NdotV{ fabsf(dot(N, V)) };
	Float NdotL{ fabsf(dot(N, L)) };
	Float NdotH{ fabsf(dot(N, H)) };

	Float FV{ schlickWeight(NdotV) };
	Float FL{ schlickWeight(NdotL) };

	Float Fss90{ mat.roughness * NdotH * NdotH };
	Float Fss{ mix(1.0f, Fss90, FL) * mix(1.0f, Fss90, FV) };
	Float Ss{ 1.25f * (Fss * (1.0f / (NdotV + NdotL) - 0.5f) + 0.5f) };

	color = disneySubsurfaceColor(mat, N, V, L, H);
	bsdf = Float3(INV_PI * Ss);
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


DEVICE void disneyPdf(Material const& mat, Float3 const& Ng, Float3 const& N, 
	Float3 const& Nb, Float3 const& T, Float3 const& B, Float3 const& V, Float3 const& L, 
	Float3 const& H, Float& pdf)
{
	pdf = 0.0f;

	Float alpha{ max(0.002f, mat.roughness * mat.roughness) };
	Float tAlpha{ max(0.002f, mat.transmissionRoughness * mat.transmissionRoughness) };
	Float aspect{ sqrtf(1.0f - mat.anisotropic * 0.9f) };
	Float2 alphaAniso{ max(0.002f, alpha / aspect), max(0.002f, alpha * aspect) };
	Float clearcoatAlpha{ mix(0.1f, MIN_ALPHA, mat.clearcoatRoughness) };

	Float diffuse{ 0.0f }, clearcoat{ 0.0f }, microfacet{ 0.0f }, microfacetTransmission{ 0.0f };
	
	pdfCosineHemisphere(L, Nb, diffuse);
	pdfGTR1(V, L, H, Nb, alpha, clearcoat);
	if (mat.anisotropic == 0.0f)
		pdfGTR2(V, L, H, Nb, alpha, microfacet);
	else
		pdfGTR2Aniso(V, L, H, Nb, T, B, alphaAniso, microfacet);

	if (mat.transmission > EPSILON && sameHemisphere(V, L, Ng))
		pdfGTR2Transmission(V, L, Nb, mat.transmissionRoughness, mat.ior, microfacetTransmission);

	Float nComp{ 3.0f };
	Float metallicKludge{ mat.metallic };
	Float transmissionKludge{ mat.transmission };
	nComp -= mix(transmissionKludge, metallicKludge, mat.metallic);
	pdf = (diffuse + microfacet + microfacetTransmission + clearcoat) / nComp;


}


DEVICE void disneyF(Material const& mat, Float3 const& Ng, Float3 const& N, Float3 const& Nb,
	Float3 const& T, Float3 const& B, Float3 const& V, Float3 const& L, Float3 const& H,
	Float3& bsdf)
{
	// 1. set bsdf to 0
	bsdf = Float3{ 0.0f };

	// 2. chck if transmissiv material (refraction)
	if (sameHemisphere(V, L, Nb) && mat.transmission > 0.0f)
	{
		// HANDLE TRANSMISSION
		return;
	}

	// 3. determin coat, sheen
	float clearcoat{ 0.0f };
	// disneyClearcoat(mat, Nb, V, L, H);

	Float3 sheen{ 0.0f };
	// disneySheen(mat, Nb, V, L, H);

	// 4. determine diffuse and subsurface color + bsdf
	Float3 diffuseBsdf{}, diffuseColor{};
	disneyDiffuse(mat, Nb, V, L, H, diffuseBsdf, diffuseColor);
	Float3 subsurfaceBsdf{}, subsurfaceColor{};
	disneySubsurface(mat, Nb, V, L, H, subsurfaceBsdf, subsurfaceColor);

	// 5. set gloss => (an)isotropic
	Float3 gloss{ 0.0f };
	if (mat.anisotropic == 0.0f)
	{
		//gloss = disneyMicroFacetIsotropic(mat, Nb, V, L, H);
	}
	else
	{
		//gloss = disneyMicroFacetAnisotropic(mat, Nb, V, L, H, T, B);
	}

	// 6. return bsdf by mixing colors
	bsdf = (mix(diffuseBsdf * diffuseColor, subsurfaceBsdf * subsurfaceColor, mat.subsurface)
		* (1.0f - mat.metallic) * (1.0f - mat.transmission) + sheen + clearcoat + gloss) *
		fabsf(dot(L, Nb));
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


DEVICE void sampleDisneyBrdf(
	Material const& mat, Random& random, Float3 const& Ng, Float3 const& N, Float3 const& Nb,
	Float3 const& T, Float3 const& B, Float3& V, Float3& L, Float& pdf, Float3& bsdf)
{
	toLocal(T, B, N, V);

	sampleCosineHemisphere({ random(), random() }, L);
	pdfCosineHemisphere(V, L, pdf);

	toWorld(T, B, N, V);
	toWorld(T, B, N, L);

	Float3 H{ computeHalfVector(V, L) };
	disneyF(mat, Ng, N, Nb, T, B, V, L, H, bsdf);
	disneyPdf(mat, Ng, N, Nb, T, B, V, L, H, pdf);
}




#endif // DISNEY_HPP