#ifndef BA_RENDERER_HPP
#define BA_RENDERER_HPP

#include <owl/owl.h>
#include <vector>
#include <owl/common/math/vec.h>
#include <pathTracer/StbUtils.hpp>

namespace ba
{
    /// <summary>
    /// TrianglesMesh is a data transfer user-defined data type.
    /// Its purpose is to transfer data of meshes to the calling system.
    /// </summary>
    struct Mesh
    {
        std::vector<owl::vec3i> index;
        std::vector<owl::vec3f> vertex;
    };

    struct Camera
    {
        owl::vec3f const lookFrom;
        owl::vec3f const lookAt;
        owl::vec3f const lookUp;
        float const cosFovy;
    };

    struct Renderer
    {
        // make context and shader
        OWLContext context;
        OWLModule module;

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

        OWLTexture environmentMap;
    };
}

#endif // !BA_RENDERER_HPP
