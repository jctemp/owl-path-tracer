#include "deviceCode.hpp"
#include <owl/owl_device.h>
#include <optix_device.h>

using namespace owl;

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
    const RayGenData& self{ getProgramData<RayGenData>() };
    const vec2i pixelID{ getLaunchIndex() };

    const vec2f screen{ (vec2f{pixelID} + vec2f{0.5f}) / vec2f{self.fbSize} };

    owl::Ray ray;
    ray.origin
        = self.camera.pos;
    ray.direction
        = normalize(self.camera.dir_00
            + screen.u * self.camera.dir_du
            + screen.v * self.camera.dir_dv);

    vec3f color;
    owl::traceRay(
        /*accel to trace against*/self.world,
        /*the ray to trace*/ray,
        /*prd*/color);

    const int fbOfs = pixelID.x + self.fbSize.x * (self.fbSize.y - 1 - pixelID.y);
    self.fbPtr[fbOfs]
        = owl::make_rgba(color);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
    vec3f& prd = owl::getPRD<vec3f>();

    const TrianglesGeomData& self = owl::getProgramData<TrianglesGeomData>();

    // compute normal:
    const int   primID = optixGetPrimitiveIndex();
    const vec3i index = self.index[primID];
    const vec3f& A = self.vertex[index.x];
    const vec3f& B = self.vertex[index.y];
    const vec3f& C = self.vertex[index.z];
    const vec3f Ng = normalize(cross(B - A, C - A));

    assert(Ng != vec3f(0));

    const vec3f rayDir = optixGetWorldRayDirection();
    prd = Ng;
    //prd = (.2f + .8f * fabs(dot(rayDir, Ng))) * self.color;
}

OPTIX_MISS_PROGRAM(miss)()
{
    const vec2i pixelID = owl::getLaunchIndex();

    const MissProgData& self = owl::getProgramData<MissProgData>();

    vec3f& prd = owl::getPRD<vec3f>();
    int pattern = (pixelID.x / 8) ^ (pixelID.y / 8);
    prd = (pattern & 1) ? self.color1 : self.color0;
}
