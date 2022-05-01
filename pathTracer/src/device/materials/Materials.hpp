#ifndef MATERIALS_INTERFACE_HPP
#define MATERIALS_INTERFACE_HPP
#pragma once

#include "../Globals.hpp"

template<Material M>
DEVICE void sampleF(MaterialStruct const& mat, Random& random, Float3 const& Ng, Float3 const& N,
	Float3 const& T, Float3 const& B, Float3 const& V, Float3& L, Float& pdf, Float3& bsdf);

template<Material M>
DEVICE void f(MaterialStruct const& mat, Float3 const& Ng, Float3 const& N,
	Float3 const& T, Float3 const& B, Float3 const& V, Float3 const& L, Float3 const& H,
	Float3& bsdf);

template<Material M>
DEVICE void pdf(MaterialStruct const& mat, Float3 const& V, Float3 const& L,
	Float& pdf);


#endif // !MATERIALS_INTERFACE_HPP
