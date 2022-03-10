#include "PathTracer.hpp"
#include "Random.hpp"

#include <owl/owl_device.h>
#include <optix_device.h>

using namespace owl;

__device__
ba::LCG<4> random{};

struct PerRayData
{
    vec3f color;
    int32_t depth;
    OptixTraversableHandle world;
};

inline __device__
vec3f toVec3f(float3 f)
{
    return vec3f{ f.x, f.y, f.z };
}

inline __device__
vec3f randomUnitSphere()
{
    while (true)
    {
        auto p = vec3f{ random(), random(), random() };
        if (dot(p, p) > 1) continue;
        return p;
    }
}

inline __device__
vec3f randomHemisphere(const vec3f& normal)
{
    vec3f in_unit_sphere = randomUnitSphere();
    if (owl::dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
    RayGenData const& self{ getProgramData<RayGenData>() };
    vec2i const pixelID{ getLaunchIndex() };

    int32_t const samplesPerPixel{ 128 };
    int32_t const pathDepth{ 64 };

    vec3f color{ 0.0f };
    for (int32_t s{ 0 }; s < samplesPerPixel; ++s)
    {
        vec2f const rand{ random(), random() };
        vec2f const screen{ (vec2f{pixelID} + rand) / vec2f{self.fbSize} };

        owl::Ray ray{ self.camera.pos, normalize(self.camera.dir_00
            + screen.u * self.camera.dir_du + screen.v * self.camera.dir_dv), 0.001f, FLT_MAX };

        PerRayData prd{ {0.0f}, pathDepth, self.world };
        owl::traceRay(
            /*accel to trace against*/self.world,
            /*the ray to trace*/ray,
            /*prd*/prd);
    
        color += 0.5f * prd.color;
    }

    color *= 1.0f / samplesPerPixel;
    // gamma correction
    color = owl::sqrt(color);

    const int fbOfs = pixelID.x + self.fbSize.x * (self.fbSize.y - 1 - pixelID.y);
    self.fbPtr[fbOfs]
        = owl::make_rgba(color);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
    PerRayData& prd = owl::getPRD<PerRayData>();

    // if the max depth is reached terminate path
    if (prd.depth == 0)
    {
        prd.color = vec3f{ 0.0f, 0.0f, 0.0f };
        return;
    }

    // get ray data:
    float const f = optixGetRayTmax();
    vec3f const O = toVec3f(optixGetWorldRayOrigin());
    vec3f const d = toVec3f(optixGetWorldRayDirection());
    vec3f const P = O + d * f;

    // get geo data:
    TrianglesGeomData const& self = owl::getProgramData<TrianglesGeomData>();

    // compute normal:
    int const primID = optixGetPrimitiveIndex();
    vec3i const index = self.index[primID];
    vec3f const& A = self.vertex[index.x];
    vec3f const& B = self.vertex[index.y];
    vec3f const& C = self.vertex[index.z];
    vec3f const N = normalize(cross(B - A, C - A));

    assert(N != vec3f{ 0 });

    // compute target + new ray
    vec3f target{ P + N + owl::normalize(randomHemisphere(N)) };

    Ray ray{ P, normalize(target - P), 0.001f, FLT_MAX };

    PerRayData nPrd{ {0.0f}, prd.depth-1, prd.world };
    owl::traceRay(
        prd.world,
        ray,
        nPrd
    );

    prd.color += 0.5f * nPrd.color;
}

OPTIX_MISS_PROGRAM(miss)()
{
    vec2i const pixelID = owl::getLaunchIndex();
    MissProgData const& self = owl::getProgramData<MissProgData>();

    vec3f const d = toVec3f(optixGetWorldRayDirection());
    auto t{ 0.5f * (d.y + 1.0f) };
    PerRayData& prd = owl::getPRD<PerRayData>();
    prd.color = (t) * vec3f{ 1.0f, 1.0f, 1.0f } + (1.0f - t) * vec3f{ 0.1f, 0.2f, 0.4f };
    prd.depth = 0;
}
