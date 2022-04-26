#ifndef MICROFACET_HPP
#define MICROFACET_HPP

#include "../SampleMethods.hpp"


namespace Microfacet
{
	namespace Distributions
	{
		struct Blinn
		{
			struct Data
			{
				Float expU, expV, exp, expUV;      /**< Internal data used for NDF calculation. */
				Float alphaU2, alphaV2;
			};

			DEVICE_STATIC Data create(Data& d, Float roughnessU, Float roughnessV)
				// http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
			{
				static const auto convertAlpha = [](Float roughness)
				{
					roughness = fmaxf(0.01f, roughness);
					return powf(roughness, 4);
				};

				static const auto convertExp = [](Float roughness)
				{
					return 2.0f / convertAlpha(roughness) - 2.0f;
				};

				d.expU = convertExp(roughnessU);
				d.expV = convertExp(roughnessV);
				d.expUV = sqrtf((d.expU + 2.0f) * (d.expV + 2.0f));
				d.exp = sqrtf((d.expU + 2.0f) / (d.expV + 2.0f));
				d.alphaU2 = convertAlpha(roughnessU);
				d.alphaV2 = convertAlpha(roughnessV);
			}

			// Distribution probability of facet with normal (h)
			DEVICE_STATIC Float D(Data const& d, Float3 const& H)
			{
				auto const NoH{ absCosTheta(H) };
				if (NoH <= 0.0f) return 0.0f;

				auto const sinPhiSqr{ sin2Phi(H) };
				auto const cosPhiSqr{ cos2Phi(H) };
				return d.expUV * powf(NoH, cosPhiSqr * d.expU + sinPhiSqr * d.expV) * INV_TWO_PI;
			}

			// shadow-masking function
			DEVICE_STATIC Float G1(Data const& d, Float3 const& V)
			{
				auto const absTan{ fabsf(tanTheta(V)) };
				if (isinf(absTan)) return 0.0f;
				const auto cos_phi_sq{ cos2Phi(V) };
				const auto a{ 1.0f / (sqrtf(cos_phi_sq * d.alphaU2 + (1.0f - cos_phi_sq) * d.alphaV2) * absTan) };
				if (a > 1.6f || isinf(a)) return 1.0f;
				return (3.535f * a + 2.181f * a * a) / (1.0f + 2.276f * a + 2.577f * a * a);
			}

			// visibility term, smith shadow masking
			DEVICE_STATIC Float G(Data const& d, Float3 const& L, Float3 const& V)
			{
				return G1(d, V) * G1(d, L);
			}

			// sampling a normal respect to given data d
			DEVICE_STATIC Float3 sampleF(Data const& d, Random& rng)
			{
				Float2 sample{ rng(), rng() };

				auto phi{ 0.0f };
				if (d.expU == d.expV) {
					phi = TWO_PI * sample.v;
				}
				else {
					static const Int offset[5]{ 0 , 1 , 1 , 2 , 2 };
					Int const i{ sample.v == 0.25f ? 0 : (int)(sample.v * 4.0f) };
					phi = atanf(d.exp * atanf(TWO_PI * sample.v)) + offset[i] * PI;
				}

				auto const sinPhiH{ sinf(phi) };
				auto const sinPhiSqr{ sinPhiH * sinPhiH };
				auto const alpha{ d.expU * (1.0f - sinPhiSqr) + d.expV * sinPhiSqr };
				auto const cosTheta{ powf(sample.u, 1.0f / (alpha + 2.0f)) };
				auto const sinTheta{ sqrtf(1.0f - pow2(cosTheta)) };

				return toSphereCoordinates(sinTheta, cosTheta, phi);
			}
		};

		struct Beckmann
		{
			struct Data
			{
				Float alphaU, alphaV;        /**< Internal data used for NDF calculation. */
				Float alphaU2, alphaV2, alphaUV, alpha;
			};

			// Create data struct for calculations.
			DEVICE_STATIC Data create(Data& d, Float roughnessU, Float roughnessV)
			{
				static const auto convert = [](Float roughness) {
					roughness = fmaxf(roughness, 0.01f);
					return roughness * roughness;
				};

				d.alphaU = convert(roughnessU);
				d.alphaV = convert(roughnessV);
				d.alphaU2 = d.alphaU * d.alphaU;
				d.alphaV2 = d.alphaV * d.alphaV;
				d.alphaUV = d.alphaU * d.alphaV;
				d.alpha =   d.alphaV / d.alphaU;
			}

			// Distribution probability of facet with normal (h)
			DEVICE_STATIC Float D(Data const& d, Float3 const& H)
			{
				auto const cosThetaSqr{ cos2Theta(H) };
				if (cosThetaSqr <= 0.0f) return 0.f;
				return exp2f((pow2(H.x) / d.alphaU2 + pow2(H.z) / d.alphaV2) / (-cosThetaSqr)) / (PI * d.alphaUV * pow2(cosThetaSqr));
			}

			// shadow-masking function
			DEVICE_STATIC Float G1(Data const& d, Float3 const& H)
			{
				const auto absTan = fabsf(tanTheta(H));
				if (isinf(absTan)) return 0.0f;
				const auto cos_phi_sq = cos2Phi(H);
				const auto a = 1.0f / (sqrtf(cos_phi_sq * d.alphaU2 + (1.0f - cos_phi_sq) * d.alphaV2) * absTan);
				if (a > 1.6f || isinf(a)) return 1.0f;
				return (3.535f * a + 2.181f * a * a) / (1.0f + 2.276f * a + 2.577f * a * a);
			}

			// visibility term, smith shadow masking
			DEVICE_STATIC Float G(Data const& d, Float3 const& L, Float3 const& V)
			{
				return G1(d, V) * G1(d, L);
			}

			// sampling a normal respect to given data d
			DEVICE_STATIC Float3 sampleF(Data const& d, Random& rng)
			{
				Float2 bs{ rng(), rng() };

				auto const logSample = log2f(bs.u);
				float theta, phi;

				if (d.alphaU == d.alphaV) 
				{
					theta = atanf(sqrtf(-1.0f * d.alphaUV * logSample));
					phi = TWO_PI * bs.v;
				}
				else 
				{
					static const int offset[5] { 0 , 1 , 1 , 2 , 2 };
					const auto i = bs.v == 0.25f ? 0 : (int)(bs.v * 4.0f);
					phi = atanf(d.alpha * tanf(TWO_PI * bs.v)) + offset[i] * PI;
					const auto sinPhi = sinf(phi);
					const auto sinPhiSqr = sinPhi * sinPhi;
					theta = atanf(sqrtf(-logSample / ((1.0f - sinPhiSqr) / d.alphaU2 + sinPhiSqr / d.alphaV2)));
				}

				return toSphereCoordinates(theta, phi);
			}
		};

		struct GGX
		{
			struct Data
			{
				Float alphaU, alphaV;        /**< Internal data used for NDF calculation. */
				Float alphaU2, alphaV2, alphaUV, alpha;
			};

			// Create data struct for calculations.
			DEVICE_STATIC Data create(Data& d, Float roughnessU, Float roughnessV)
			{
				static const auto convert = [](Float roughness) {
					roughness = fmaxf(roughness, 0.001f);
					return pow2(roughness);
				};

				d.alphaU = convert(roughnessU);
				d.alphaV = convert(roughnessV);
				d.alphaU2 = d.alphaU * d.alphaU;
				d.alphaV2 = d.alphaV * d.alphaV;
				d.alphaUV = d.alphaU * d.alphaV;
				d.alpha = d.alphaV / d.alphaU;
			}

			// Distribution probability of facet with normal (h)
			DEVICE_STATIC Float D(Data const& d, Float3 const& H)
			{
				auto const cosThetaSqr{ cos2Theta(H) };
				if (cosThetaSqr <= 0.0f) return 0.f;
				auto const beta{ (cosThetaSqr + (pow2(H.x) / d.alphaU2 + pow2(H.z) / d.alphaV2)) };
				return 1.0f / (PI * d.alphaUV * beta * beta);
			}

			// shadow-masking function
			DEVICE_STATIC Float G1(Data const& d, Float3 const& H)
			{
				auto const tanThetaSqr = tan2Theta(H);
				if (isinf(tanThetaSqr)) return 0.0f;
				auto const cosPhiSqr = cos2Phi(H);
				auto const alpha2 = cosPhiSqr * d.alphaU2 + (1.0f - cosPhiSqr) * d.alphaV2;
				return 2.0f / (1.0f + sqrtf(1.0f + alpha2 * tanThetaSqr));
			}

			// visibility term, smith shadow masking
			DEVICE_STATIC Float G(Data const& d, Float3 const& L, Float3 const& V)
			{
				return G1(d, V) * G1(d, L);
			}

			// sampling a normal respect to given data d
			DEVICE_STATIC Float3 sampleF(Data const& d, Random& rng)
			{
				Float2 bs{ rng(), rng() };

				Float theta, phi;
				if (d.alphaU == d.alphaV)
				{
					theta = atanf(d.alphaU * sqrtf(bs.v / (1.0f - bs.v)));
					phi = TWO_PI * bs.u;
				}
				else
				{
					static const int offset[5] = { 0 , 1 , 1 , 2 , 2 };
					const auto i = bs.v == 0.25f ? 0 : (int)(bs.v * 4.0f);
					phi = atanf(d.alpha * std::tan(TWO_PI * bs.v)) + offset[i] * PI;
					auto const sin_phi = sinf(phi);
					auto const sin_phi_sq = sin_phi * sin_phi;
					auto const cos_phi_sq = 1.0f - sin_phi_sq;
					float beta = 1.0f / (cos_phi_sq / d.alphaU2 + sin_phi_sq / d.alphaV2);
					theta = atanf(sqrtf(beta * bs.u / (1.0f - bs.u)));
				}
				return toSphereCoordinates(theta, phi);
			}
		};
	}

	DEVICE Float schlickWeight(Float cos) {
		return powf(saturate(1.0f - cos), 5);
	}

	DEVICE Float3 schlickFresnel(const Float3& F0, Float cos) {
		return F0 + schlickWeight(cos) * (Float3(1.0f) - F0);
	}

	template<typename T = Distributions::GGX>
	DEVICE Float3 f(ShadingData& sd, Float3 const& V, Float3 const& L, T::Data const &data)
	{
		if (!sameHemisphere(V, L)) return Float3{ 0.0f };
		if (V.z < 0.0f) return Float3{ 0.0f };

		auto const NoV{ absCosTheta(V) };
		if (NoV == 0.0f || NoL == 0.0f) return Float3{ 0.0f };

		auto const H{ normalize(L + V) };
		auto const F{ schlickFresnel({1.5f}, dot(L, V)) };

		//TODO: do proper reflectance
		return Float3{ 0.8f } * T::D(data, H) * T::G(data, L, V) / (4.0f * NoV);
	}

	template<typename T = Distributions::GGX>
	DEVICE void pdf(Float3 const& V, Float3 const& L, Float& mPdf, T::Data const& data)
	{
		mPdf = 0.0f;
		if (!sameHemisphere(V, L)) return;
		if (V.z < 0.0f) return;

		auto const H{ normalize(L + V)};
		auto const VoH{ fabsf(dot(V, H)) };
		mPdf = T::D(data, H) * absCosTheta(H);
	}

	template<typename T = Distributions::GGX>
	DEVICE Float3 sampleF(ShadingData& sd, Float3 const& V, Float3& L, Float& mPdf)
	{
		T::Data data{};
		T::create(data, sd.md->roughness, sd.md->roughness);

		auto const H{ T::sampleF(data, sd.random) };
		L = reflect(V, H);
		pdf<T>(V, L, mPdf, data);
		
		if (!sameHemisphere(V, L)) return Float3{ 0.0f };
		if (V.z < 0.0f) return Float3{ 0.0f };

		return f<T>(sd, V, L, data);
	}


}


#endif // !MICROFACET_HPP