#ifndef DEVICE_HPP
#define DEVICE_HPP
#pragma once

#include "../shared/shared.hpp"

#define GET(RETURN, TYPE, BUFFER, ADDRESS)\
{\
if (BUFFER.data == nullptr) {\
	::printf("Device Error (%d, %s): buffer was nullptr.\n", __LINE__, __FILE__); asm("trap;");}\
if (ADDRESS >= BUFFER.count) {\
	::printf("Device Error (%d): out of bounds access (address: %d, size %d).\n",\
	__LINE__, ADDRESS, uint32_t(BUFFER.count)); asm("trap;");} \
RETURN = ((TYPE*)BUFFER.data)[ADDRESS];\
}

#define ASSERT(check, msg) \
if (check) {::printf("Device Error (%d, %s): %s\n", __LINE__, __FILE__, msg); asm("trap;"); }

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

enum class ScatterEvent
{
	BOUNCED = 1 << 0,
	CANCELLED = 1 << 1,
	MISS = 1 << 2,
	NONE = 1 << 3
};

struct InterfaceStruct
{
	/* triangle points */
	vec3 TRI[3];

	/* hit position */
	vec3 P;

	/* shading normal */
	vec3 N;

	/* geometric normal */
	vec3 Ng;

	/* view direction (wo or V) */
	vec3 V;

	/* barycentrics */
	vec2 UV;

	/* thit */
	float t;

	/* primitive id => 0 if not exists */
	int32_t prim;

	/* material id for LP reference */
	int32_t matId;

	/* light id for LP reference */
	int32_t lightId;
};

struct per_ray_data
{
	Random& random;
	ScatterEvent scatterEvent;
	InterfaceStruct* is;
	material_data* ms;
};

#endif // !DEVICE_HPP
