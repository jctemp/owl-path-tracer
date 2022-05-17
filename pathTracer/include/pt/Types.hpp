#ifndef TYPES_HPP
#define TYPES_HPP

#include <device/device.hpp>
#include <vector>

struct entity
{
    Int materialId{ -1 };
    Int lightId{ -1 };
};

struct Camera
{
    Int2 const fbSize;
    Float3 const lookFrom;
    Float3 const lookAt;
    Float3 const lookUp;
    float const vfov;
};


struct Renderer
{
    // make context and shader
    OWLContext context;
    OWLModule module;

    // launchParams static accessible mem.
    OWLLaunchParams launchParams;
    uint32_t maxDepth;
    uint32_t samplesPerPixel;

    // Programs
    OWLRayGen rayGenenration;
    OWLMissProg missProg;

    // link between host and device
    OWLBuffer frameBuffer;

    // Geometry and mesh
    OWLGeomType trianglesGeomType;
    std::vector<OWLGeom> geoms;
    bool requireBuild;

    // Group to handle geometry
    OWLGroup world;

    // Texture holding env. information
    OWLTexture environmentMap;
    bool useEnvironmentMap;
};


#endif // !TYPES_HPP
