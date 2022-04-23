#ifndef BA_RENDERER_HPP
#define BA_RENDERER_HPP

#include <owl/owl.h>
#include <vector>
#include <owl/common/math/vec.h>
#include <pathTracer/StbUtils.hpp>


/// <summary>
/// TrianglesMesh is a data transfer user-defined data type.
/// Its purpose is to transfer data of meshes to the calling system.
/// </summary>
struct Mesh
{
    std::vector<owl::vec3i> index;
    std::vector<owl::vec3f> vertex;
    std::vector<owl::vec3f> normal;
};

struct Camera
{
    owl::vec3f const lookFrom;
    owl::vec3f const lookAt;
    owl::vec3f const lookUp;
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


#endif // !BA_RENDERER_HPP
