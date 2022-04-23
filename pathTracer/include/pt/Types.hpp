#ifndef TYPES_HPP
#define TYPES_HPP

#include <device/Globals.hpp>
#include <vector>


struct Mesh
{
    std::vector<Int3> index;
    std::vector<Float3> vertex;
    std::vector<Float3> normal;
};


struct Camera
{
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
    OWLRayGen rayGen;
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
