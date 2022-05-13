#ifndef DEVICE_GLOBALS_HPP
#define DEVICE_GLOBALS_HPP
#pragma once

#include "Shared.hpp"
#include "Random.hpp"

using Random = LCG<4>;

#define DEVICE_STATIC static __owl_device
#define DEVICE_INL inline __owl_device
#define DEVICE __owl_device

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
	Float3 TRI[3];

	/* hit position */
	Float3 P;

	/* shading normal */
	Float3 N;

	/* geometric normal */
	Float3 Ng;

	/* view direction (wo or V) */
	Float3 V;

	/* barycentrics */
	Float2 UV;

	/* thit */
	Float t;

	/* primitive id => 0 if not exists */
	Int prim;

	/* material id for LP reference */
	Int matId;

	/* light id for LP reference */
	Int lightId;
};

struct PerRayData
{
	Random& random;
	ScatterEvent scatterEvent;
	InterfaceStruct* is;
	MaterialStruct* ms;
};



#endif // !DEVICE_GLOBALS_HPP
