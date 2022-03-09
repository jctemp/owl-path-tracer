#pragma once

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#ifndef M_PI_2
#define M_PI_2 (2.f * M_PI)
#endif

#ifndef M_1_PI
#define M_1_PI 0.318309886183790671538f
#endif

#define EPSILON 0.0001f
#define SMALL_EPSILON 0.00000000001f
#define MIN_ROUGHNESS .04f
#define MIN_ALPHA MIN_ROUGHNESS * MIN_ROUGHNESS

#define PRIMARY_RAY 0
#define OCCLUSION_RAY 1
#define NUM_RAY_TYPES 2
// #define MAX_PATH_DEPTH 50

#ifndef FLT_MIN
#define FLT_MIN 1.175494e-38
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.402823e+38
#endif