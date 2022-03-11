#ifndef DEVICE_CODE_HPP
#define DEVICE_CODE_HPP

#include <owl/owl.h>
#include <owl/common/math/LinearSpace.h>
#include <owl/common/math/vec.h>
#include <cuda_runtime.h>

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
    OptixTraversableHandle world;

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
    owl::vec3f color0;
    owl::vec3f color1;
    
    // for textures
    cudaTextureObject_t envMap;
};

#endif // DEVICE_CODE_HPP