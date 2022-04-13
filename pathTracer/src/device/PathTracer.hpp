#ifndef DEVICE_CODE_HPP
#define DEVICE_CODE_HPP

#include <owl/owl.h>
#include <owl/common/math/LinearSpace.h>
#include <owl/common/math/vec.h>
#include <cuda_runtime.h>

struct LaunchParams
{
    uint32_t maxDepth;
    uint32_t samplesPerPixel;
    OptixTraversableHandle world;
    cudaTextureObject_t environmentMap;
};

struct TrianglesGeomData
{
    owl::vec3f color;
    owl::vec3i* index;
    owl::vec3f* vertex;
};

struct RayGenData
{
    uint32_t* fbPtr;
    owl::vec2i fbSize;

    struct 
    {
        owl::vec3f pos;
        owl::vec3f dir_00;
        owl::vec3f dir_du;
        owl::vec3f dir_dv;
    } camera;
};

struct MissProgData
{
};

#endif // DEVICE_CODE_HPP