#ifndef TYPES_HPP
#define TYPES_HPP

#include <device/device.hpp>
#include <vector>

struct entity
{
    Int materialId{ -1 };
    Int lightId{ -1 };
};

struct optix_data
{
    // make context and shader
    OWLContext context;
    OWLModule module;

    // launchParams static accessible mem.
    OWLLaunchParams launch_params;
    uint32_t max_path_depth;
    uint32_t max_samples;

    // Programs
    OWLRayGen ray_gen_program;
    OWLMissProg miss_program;

    // link between host and device
    glm::ivec2 buffer_size{ 1024 };
    OWLBuffer frame_buffer;

    // Geometry and mesh
    OWLGeomType triangle_geom;
    std::vector<OWLGeom> geoms;

    // Group to handle geometry
    OWLGroup world;

    // Texture holding env. information
    OWLTexture environment_map;
    bool use_environment_map;
};

static optix_data od{};

#endif // TYPES_HPP
