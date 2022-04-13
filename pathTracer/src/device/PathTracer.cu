#include "PathTracer.hpp"
#include "Random.hpp"
#include "Types.hpp"
#include "Sampling.hpp"
#include "Bxdf.hpp"

#include <owl/owl_device.h>
#include <optix_device.h>

#define MAX_DEPTH 64
#define SAMPLES_PER_PIXEL 128
#define T_MIN 1e-3f
#define T_MAX 1e10f

using namespace owl;

__constant__ LaunchParams optixLaunchParams;
__device__ ba::LCG<4> random{};

struct Intersection
{
    owl::vec3f normal;
    owl::vec3f point;
    owl::vec3f wo;
    float t;
};

struct PerRayData
{
    Intersection si;
};


inline __device__ vec2f uvOnSphere(vec3f n)
{
    auto u = 0.5f + atan2(n.x, n.z) / (2 * M_PI);
    auto v = 0.5f + asin(n.y) / M_PI;
    return vec2f{ u,v };
}

inline __device__ vec3f makeVec3f(float3 f)
{
    return vec3f{ f.x, f.y, f.z };
}

inline __device__ vec3f randomUnitSphere()
{
    while (true)
    {
        auto p = vec3f{ random(), random(), random() };
        if (dot(p, p) > 1) continue;
        return p;
    }
}

inline __device__ vec3f randomHemisphere(const vec3f& normal)
{
    vec3f in_unit_sphere = randomUnitSphere();
    if (owl::dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

inline __device__ owl::vec3f tracePath(owl::Ray &ray, PerRayData& prd)
{
    owl::vec3f attenuation{ 1.0f };
    
    for (int32_t depth{ 0 }; depth < optixLaunchParams.maxDepth; ++depth)
    {
        // 1) find intersection
        owl::traceRay(
            /* accel to trace against */ optixLaunchParams.world,
            /* the ray to trace       */ ray,
            /* prd                    */ prd);


        // 2) terminate if miss => sample background
        if (prd.si.t < 0)
        {
            if (false && optixLaunchParams.environmentMap)
            {
                vec2f tc{ uvOnSphere(ray.direction) };
                owl::vec4f const texColor{ 
                    tex2D<float4>(optixLaunchParams.environmentMap, tc.x, tc.y) };
                return vec3f{ texColor } *attenuation;
            }
            else
            {
                return vec3f{ 0.6f, 0.8f, 1.0f } *attenuation;
            }
        }

        owl::vec3f wi{ 0.0f };
        float pdf{ 0.0f };
        ba::cosineSampleHemisphere(prd.si.normal, random(), random(), wi, pdf);

        if (pdf < 0) return { 0.0f };

        // new ray
        attenuation *= 0.8f;
        ray = owl::Ray{
            prd.si.point,
            wi,
            T_MIN, T_MAX
        };

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        // 3) load material

        // 4) terminate if hit light => sample light

        // 5) sample bsdf

        // 6) sample light source

        // 7) accum radiance

        // 8) russian roulette



    }

    return { 0.0f };
}

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
    RayGenData const& self{ getProgramData<RayGenData>() };
    vec2i const pixelID{ getLaunchIndex() };

    PerRayData prd{ {{0.0f}, {0.0f}, {0.0f}, 0.0f} };
    vec3f color{ 0.0f };

    for (int32_t s{ 0 }; s < optixLaunchParams.samplesPerPixel; ++s)
    {
        // shot ray with slight randomness to make soft edges
        vec2f const rand{ random(), random() };
        vec2f const screen{ (vec2f{pixelID} + rand) / vec2f{self.fbSize} };

        // determine initial ray form the camera
        owl::Ray ray{ self.camera.pos, normalize(self.camera.dir_00
            + screen.u * self.camera.dir_du + screen.v * self.camera.dir_dv), 0.001f, FLT_MAX };
    
        color += tracePath(ray, prd);
    }

    // take the average of all samples per pixel and apply gamma correction
    color *= 1.0f / optixLaunchParams.samplesPerPixel;
    color = owl::sqrt(color);

    // save result into the buffer
    const int fbOfs = pixelID.x + self.fbSize.x * (self.fbSize.y - 1 - pixelID.y);
    self.fbPtr[fbOfs]
        = owl::make_rgba(color);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
    PerRayData& prd{ owl::getPRD<PerRayData>() };
    Intersection& si{ prd.si };

    // barycentrics
    float b1{ optixGetTriangleBarycentrics().x };
    float b2{ optixGetTriangleBarycentrics().y };
    float b0{ 1 - b1 - b2 };

    // get direction
    vec3f const direction{ makeVec3f(optixGetWorldRayDirection()) };

    // get geometric data:
    TrianglesGeomData const& self = owl::getProgramData<TrianglesGeomData>();
    uint32_t const primID{ optixGetPrimitiveIndex() };
    vec3i const index{ self.index[primID] };
    vec3f const& p0{ self.vertex[index.x] };
    vec3f const& p1{ self.vertex[index.y] };
    vec3f const& p2{ self.vertex[index.z] };

    // set hit information
    si.normal = owl::normalize(owl::cross(p1 - p0, p2 - p0));
    si.point = p0 * b0 + p1 * b1 + p2 * b2;
    si.wo = -direction;
    si.t = optixGetRayTmax();
    assert(si.normal != vec3f{ 0 });
}

OPTIX_MISS_PROGRAM(miss)()
{
    PerRayData& prd = owl::getPRD<PerRayData>();
    Intersection& si{ prd.si };
    si.t = -1;
}
