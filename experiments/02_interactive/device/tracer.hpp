#pragma once

#ifndef TRACER_HPP
#define TRACER_HPP

#include <owl/owl.h>
#include <owl/common/math/vec.h>

struct RayGenData
{
	uint32_t* fbPtr;
	owl::vec2i fbSize;
	OptixTraversableHandle world;

	struct
	{
		owl::vec3f pos;
		owl::vec3f dir_00;
		owl::vec3f dir_du;
		owl::vec3f dir_dv;
	} camera;

};

struct TrianglesGeomData
{
	owl::vec3i* index;
	owl::vec3f* vertex;
	owl::vec3f color;
};

struct MissData
{
	owl::vec3f color0;
	owl::vec3f color1;
};

#endif // !TRACER_HPP

