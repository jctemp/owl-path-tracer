#ifndef MISS_HPP
#define MISS_HPP
#pragma once

#include "Globals.hpp"
#include "materials/Lambert.hpp"
#include "materials/Disney.hpp"

#include <owl/owl_device.h>
#include <optix_device.h>

using namespace owl;

OPTIX_MISS_PROGRAM(miss)()
{
	PerRayData& prd{ getPRD<PerRayData>() };
	prd.scatterEvent = ScatterEvent::MISS;
}

#endif // !MISS_HPP
