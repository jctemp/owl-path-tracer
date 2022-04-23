#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include <owl/owl.h>
#include <owl/common/math/LinearSpace.h>
#include <owl/owl_device_buffer.h>
#include <owl/common/math/vec.h>
#include <cuda_runtime.h>
#include <array>

#include "Random.hpp"

using Float = float;
using Float2 = owl::vec2f;
using Float3 = owl::vec3f;
using Float4 = owl::vec4f;
using Int = int32_t;
using Int2 = owl::vec2i;
using Int3 = owl::vec3i;
using Uint = uint32_t;
using Uint2 = owl::vec2ui;
using Uint3 = owl::vec3ui;
using Buffer = owl::device::Buffer;
using Random = LCG<4>;


#define GET(RETURN, TYPE, BUFFER, ADDRESS) \
if (BUFFER.data == nullptr) {::printf("Device Error (%d): buffer was nullptr.\n", __LINE__); asm("trap;");} \
if (ADDRESS >= BUFFER.count) {::printf("Device Error (%d): out of bounds access (address: %d, size %d).\n", __LINE__, ADDRESS, uint32_t(BUFFER.count)); asm("trap;");} \
RETURN = ((TYPE*)BUFFER.data)[ADDRESS];


#define PI           3.14159265358979323f
#define TWO_PI       6.28318530717958648f
#define PI_OVER_TWO  1.57079632679489661f 
#define PI_OVER_FOUR 0.78539816339744830f 
#define INV_PI       0.31830988618379067f 
#define INV_TWO_PI   0.15915494309189533f 
#define INV_FOUR_PI  0.07957747154594766f 
#define EPSILON      1E-5f
#define T_MIN        1E-3f
#define T_MAX        1E10f
#define DEVICE inline __owl_device


enum class ScatterEvent
{
    BOUNCED = 1 << 0,
    CANCELLED = 1 << 1,
    MISS = 1 << 2,
    NONE = 1 << 3
};

struct Material
{
    enum class Type
    {
        LAMBERT = 1 << 1,
        METAL = 1 << 2,
        DIELECTRICS = 1 << 3,
        LIGHT = 1 << 4,
    };

    Type type;
    Float3 baseColor;
    Float ior;
    Float3 emit;
};


struct ObjectData
{
    /* hit position */
    Float3 P;

    /* shading normal */
    Float3 N;

    /* geometric normal */
    Float3 Ng;

    /* view direction (wo or V) */
    Float3 V;

    /* barycentrics */
    Float2 uv;

    /* primitive id => 0 if not exists */
    Int prim;

    /* material id for LP reference */
    Uint matId;
};


struct ShaderData
{
    ObjectData* od;
    Material* md;
};


struct PerRayData
{
    Random random;
    ObjectData* od;
    ScatterEvent scatterEvent;
};


struct LaunchParams
{
    Int maxDepth;
    Int samplesPerPixel;
    Buffer materials;
    OptixTraversableHandle world;
    cudaTextureObject_t environmentMap;
    bool useEnvironmentMap;
};


struct TrianglesGeomData
{
    Uint matId;
    Int3* index;
    Float3* vertex;
    Float3* normal;
};


struct RayGenData
{
    Uint* fbPtr;
    Int2 fbSize;

    struct 
    {
        Float3 origin;
        Float3 llc;
        Float3 horizontal;
        Float3 vertical;
    } camera;
};


struct MissProgData
{
};

#endif // GLOBALS_HPP