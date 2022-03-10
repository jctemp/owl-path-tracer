#include "PathTracer.hpp"
#include "Random.hpp"

#include <owl/owl_device.h>
#include <optix_device.h>

using namespace owl;

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
    ba::LCG<4> random{};

    RayGenData const& self{ getProgramData<RayGenData>() };
    vec2i const pixelID{ getLaunchIndex() };

    random.init(pixelID.x, pixelID.y);
    int32_t const samplesPerPixel{ 128 };

    vec3f color{ 0.0f };
    for (int32_t s{ 0 }; s < samplesPerPixel; ++s)
    {
        vec2f const rand{ random(), random() };
        vec2f const screen{ (vec2f{pixelID} + rand) / vec2f{self.fbSize} };

        owl::Ray ray{ self.camera.pos, normalize(self.camera.dir_00
            + screen.u * self.camera.dir_du + screen.v * self.camera.dir_dv), 0.0001f, FLT_MAX };

        vec3f scolor;
        owl::traceRay(
            /*accel to trace against*/self.world,
            /*the ray to trace*/ray,
            /*prd*/scolor);
    
        color += scolor;
    }

    color *= 1.0f / samplesPerPixel;

    const int fbOfs = pixelID.x + self.fbSize.x * (self.fbSize.y - 1 - pixelID.y);
    self.fbPtr[fbOfs]
        = owl::make_rgba(color);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
    vec3f& prd = owl::getPRD<vec3f>();

    const TrianglesGeomData& self = owl::getProgramData<TrianglesGeomData>();

    // compute normal:
    int   const  primID = optixGetPrimitiveIndex();
    vec3i const  index = self.index[primID];
    vec3f const & A = self.vertex[index.x];
    vec3f const & B = self.vertex[index.y];
    vec3f const & C = self.vertex[index.z];
    vec3f const  Ng = normalize(cross(B - A, C - A));

    assert(Ng != vec3f(0));

    vec3f const rayDir = optixGetWorldRayDirection();
    prd = Ng;
    //prd = (.2f + .8f * fabs(dot(rayDir, Ng))) * self.color;
}

OPTIX_MISS_PROGRAM(miss)()
{
    vec2i const pixelID = owl::getLaunchIndex();
    MissProgData const& self = owl::getProgramData<MissProgData>();

    vec3f& prd = owl::getPRD<vec3f>();
    int pattern = (pixelID.x / 8) ^ (pixelID.y / 8);
    prd = (pattern & 1) ? self.color1 : self.color0;
}
