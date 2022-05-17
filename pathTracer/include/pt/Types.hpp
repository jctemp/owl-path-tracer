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
    Float3 const lookFrom;
    Float3 const lookAt;
    Float3 const lookUp;
    float const vfov;
};


struct optix_data
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
    glm::ivec2 buffer_size{ 1024 };
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

static optix_data od{};

#endif // TYPES_HPP
