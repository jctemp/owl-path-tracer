#ifndef MATERIALS_INTERFACE_HPP
#define MATERIALS_INTERFACE_HPP
#pragma once

#include "../Globals.hpp"

template<Material M>
DEVICE void sampleF(MaterialStruct& ms, Float3 const& V, Float2 u, Float3& L,
	Float3& brdf, Float& pdf);

template<Material M>
DEVICE void f(MaterialStruct& ms, Float3 const& V, Float3 const& L,
	Float3& brdf);

template<Material M>
DEVICE void pdf(Float3 const& V, Float3 const& L,
	Float& pdf);


#endif // !MATERIALS_INTERFACE_HPP
