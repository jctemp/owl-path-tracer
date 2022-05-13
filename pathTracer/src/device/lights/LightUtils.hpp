#ifndef LIGHT_UTILS_HPP
#define LIGHT_UTILS_HPP
#pragma once

#include "../Sampling.hpp"

DEVICE_INL sampleTriangle(Float2 rand)
{
	Float2 p{};
	sampleUniformTriangle(rand, p);
}

#endif // !LIGHT_UTILS_HPP
