#ifndef MISS_HPP
#define MISS_HPP
#pragma once

#include "device.hpp"
#include "materials/Lambert.hpp"
#include "materials/Disney.hpp"

#include "owl/include/owl/owl_device.h"
#include <optix_device.h>

using namespace owl;

OPTIX_MISS_PROGRAM(miss)()
{
	per_ray_data& prd{ getPRD<per_ray_data>() };
	prd.scatterEvent = ScatterEvent::MISS;
}

#endif // !MISS_HPP