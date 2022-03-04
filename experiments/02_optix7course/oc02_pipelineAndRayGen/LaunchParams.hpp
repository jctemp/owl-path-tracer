#pragma

#include <owl/common/math/vec.h>

struct LaunchParams
{
	int frameID{ 0 };
	uint32_t* colorBuffer;
	owl::vec2i fbSize;
};
