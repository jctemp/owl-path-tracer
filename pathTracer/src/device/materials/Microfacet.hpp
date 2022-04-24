#ifndef MICROFACET_HPP
#define MICROFACET_HPP

#include "../SampleMethods.hpp"

namespace Microfacet
{
	namespace Distributions
	{
		namespace Blinn
		{
			struct Data
			{
				Float expU, expV, exp, expUV;      /**< Internal data used for NDF calculation. */
				Float alphaU2, alphaV2;
			};

			// Create data struct for calculations.
			Data Blinn(Float roughnessU, Float roughnessV);

			// Distribution probability of facet with normal (h)
			Float D(Data const& d, Float3 const& H);

			// visibility term, smith shadow masking
			Float G(Data const& d, Float3 const& L, Float3 const& V);

			// shadow-masking function
			Float G1(Data const& d, Float3 const& H);

			// sampling a normal respect to given data d
			Float3 sampleF(Data const& d, Random const& rng);
		}

		namespace Beckmann
		{
			struct Data
			{
				Float alphaU, alphaV;        /**< Internal data used for NDF calculation. */
				Float alphaU2, alphaV2, alphaUV, alpha;
			};

			// Create data struct for calculations.
			Data Beckmann(Float roughnessU, Float roughnessV);

			// Distribution probability of facet with normal (h)
			Float D(Data const& d, Float3 const& H);

			// visibility term, smith shadow masking
			Float G(Data const& d, Float3 const& L, Float3 const& V);

			// shadow-masking function
			Float G1(Data const& d, Float3 const& H);

			// sampling a normal respect to given data d
			Float3 sampleF(Data const& d, Random const& rng);
		}

		namespace GGX
		{
			struct Data
			{
				Float alphaU, alphaV;        /**< Internal data used for NDF calculation. */
				Float alphaU2, alphaV2, alphaUV, alpha;
			};

			// Create data struct for calculations.
			Data GXX(Float roughnessU, Float roughnessV);

			// Distribution probability of facet with normal (h)
			Float D(Data const& d, Float3 const& H);

			// visibility term, smith shadow masking
			Float G(Data const& d, Float3 const& L, Float3 const& V);

			// shadow-masking function
			Float G1(Data const& d, Float3 const& H);

			// sampling a normal respect to given data d
			Float3 sampleF(Data const& d, Random const& rng);
		}
	}


	DEVICE Float3 f(ShadingData& sd, Float3 const& V, Float3 const& L);

	DEVICE Float3 sampleF(ShadingData& sd, Float3 const& V, Float3& L, Float& pdf);

	DEVICE void pdf(Float3 const& V, Float3 const& L, Float& pdf);

}


#endif // !MICROFACET_HPP