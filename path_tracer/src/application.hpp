
#ifndef APPLICATION_HPP
#define APPLICATION_HPP

#include "utils/mesh_loader.hpp"
#include "utils/image_buffer.hpp"
#include "owl.hpp"
#include "device/device_global.hpp"

#include <vector>
#include <string>
#include <tuple>
#include <memory>

struct entity
{
    mesh* mesh_ptr;
    int32_t materialId{ -1 };
};

struct owl_data
{
    OWLContext owl_context;
    OWLModule owl_module;

    geom_type triangle_geom;

    group world;

    ray_gen_program ray_gen_prog;
    miss_program miss_prog;
    miss_program miss_shadow_prog;

    launch_params lp;
};

struct program_data
{
    ivec2 buffer_size;
    int32_t max_samples;
    int32_t max_path_depth;

    bool environment_use;
    bool environment_auto;
    vec3 environment_color;
    float environment_intensity;
    image_buffer environment_map;

    camera_data camera;
    
    std::vector<std::tuple<std::string, material_data>> materials;
    std::vector<std::tuple<std::string, std::shared_ptr<mesh>>> meshes;
    std::vector<entity> entities;

    std::vector<geom> geoms{};
    std::vector<buffer> indices_buffer_list{};
    std::vector<buffer> vertices_buffer_list{};
    std::vector<buffer> normals_buffer_list{};

    buffer material_buffer;
    buffer vertices_buffer;
    buffer indices_buffer;
    buffer normals_buffer;
};

void init_owl_data(owl_data& data);

void init_owl_world(owl_data& data, std::vector<geom>& geoms);

void init_program_data(program_data& data, std::string assets_path, std::string scene);

#endif APPLICATION_HPP
