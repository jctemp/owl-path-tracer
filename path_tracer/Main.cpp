﻿
#include "utils/image_buffer.hpp"
#include "utils/mesh_loader.hpp"
#include "utils/parser.hpp"

#include "owl.hpp"
#include "camera.hpp"
#include "device/device_global.hpp"

#include <fmt/core.h>
#include <fmt/color.h>

#include <filesystem>

using namespace owl;

extern "C" char device_ptx[];

struct entity
{
    mesh* mesh_ptr;
    int32_t materialId{-1};
    int32_t lightId{-1};
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
    OWLMissProg miss_shadow_program;

    // link between host and device
    ivec2 buffer_size{1024};
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


void optix_init()
{
    /* ━━━━━━━━━ CREATE CONTEXT AND MODULE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
    od.context = create_context(nullptr, 1);
    od.module = create_module(od.context, device_ptx);


    /* ━━━━━━━━━ CREATE RAY GEN PROGRAM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
    var_decl ray_gen_vars
            {
                    {"fbPtr",             OWL_BUFPTR, OWL_OFFSETOF(ray_gen_data, fbPtr)},
                    {"fbSize",            OWL_INT2,   OWL_OFFSETOF(ray_gen_data, fbSize)},
                    {"camera.origin",     OWL_FLOAT3, OWL_OFFSETOF(ray_gen_data, camera.origin)},
                    {"camera.llc",        OWL_FLOAT3, OWL_OFFSETOF(ray_gen_data, camera.llc)},
                    {"camera.horizontal", OWL_FLOAT3, OWL_OFFSETOF(ray_gen_data, camera.horizontal)},
                    {"camera.vertical",   OWL_FLOAT3, OWL_OFFSETOF(ray_gen_data, camera.vertical)},
                    {nullptr}
            };

    od.ray_gen_program = create_ray_gen_program(od.context, od.module, "ray_gen", sizeof(ray_gen_data), ray_gen_vars);


    /* ━━━━━━━━━ CREATE MISS PROGRAMS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
    od.miss_program = create_miss_program(od.context, od.module, "miss", 0u, nullptr);
    owlMissProgSet(od.context, 0, od.miss_program);

    od.miss_shadow_program = create_miss_program(od.context, od.module, "miss_shadow", 0u, nullptr);
    owlMissProgSet(od.context, 1, od.miss_shadow_program);


    /* ━━━━━━━━━ CREATE LAUNCH PARAMS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
    var_decl launchParamsVars
            {
                    {"max_path_depth", OWL_USER_TYPE(uint32_t), OWL_OFFSETOF(launch_params_data, max_path_depth)},
                    {"max_samples",    OWL_USER_TYPE(uint32_t), OWL_OFFSETOF(launch_params_data, max_samples)},
                    {"material_buffer",     OWL_BUFFER,         OWL_OFFSETOF(launch_params_data, material_buffer)},
                    {"light_buffer",        OWL_BUFFER,         OWL_OFFSETOF(launch_params_data, light_buffer)},
                    {"world",               OWL_GROUP,          OWL_OFFSETOF(launch_params_data, world)},
                    {"environment_map",     OWL_TEXTURE,        OWL_OFFSETOF(launch_params_data, environment_map)},
                    {"use_environment_map", OWL_BOOL,           OWL_OFFSETOF(launch_params_data, use_environment_map)},
                    {nullptr}
            };

    od.launch_params = create_launch_params(od.context, sizeof(launch_params_data), launchParamsVars);


    /* ━━━━━━━━━ CREATE TRIANGLE GEOM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
    var_decl triangles_geom_vars
            {
                    {"matId",   OWL_INT,    OWL_OFFSETOF(triangle_geom_data, matId)},
                    {"lightId", OWL_INT,    OWL_OFFSETOF(triangle_geom_data, lightId)},
                    {"index",   OWL_BUFPTR, OWL_OFFSETOF(triangle_geom_data, index)},
                    {"vertex",  OWL_BUFPTR, OWL_OFFSETOF(triangle_geom_data, vertex)},
                    {"normal",  OWL_BUFPTR, OWL_OFFSETOF(triangle_geom_data, normal)},
                    {nullptr}
            };

    od.triangle_geom = create_geom_type(od.context, OWL_GEOM_TRIANGLES,
            sizeof(triangle_geom_data), triangles_geom_vars);

    geom_type_closest_hit_program(od.triangle_geom, od.module, "triangle_hit", 0);
}

void optix_destroy()
{
    destroy_context(od.context);
}

void optix_set_environment_map(image_buffer const& texture)
{
    if (!texture.buffer)
        return;

    if (od.environment_map != nullptr)
        destroy_texture(od.environment_map);

    od.environment_map = create_texture(
            od.context,
            {texture.width, texture.height},
            texture.buffer
    );
}

void add(entity e)
{
    mesh& mesh{*e.mesh_ptr};

    auto& vertices{mesh.vertices};
    auto& indices{mesh.indices};
    auto& normals{mesh.normals};

    // set geometry in the buffers of the object
    buffer vertex_buffer{
            create_device_buffer(od.context, OWL_FLOAT3, vertices.size(), vertices.data())};

    buffer normal_buffer{
            create_device_buffer(od.context, OWL_FLOAT3, normals.size(), normals.data())};

    buffer index_buffer{
            create_device_buffer(od.context, OWL_INT3, indices.size(), indices.data())};

    // prepare mesh for device
    geom geom_data{
            owlGeomCreate(od.context, od.triangle_geom)};

    // set specific vertex/index buffer => required for build the accel.
    set_triangle_vertices(geom_data, vertex_buffer,
            vertices.size(), sizeof(owl::vec3f));

    set_triangle_indices(geom_data, index_buffer,
            indices.size(), sizeof(ivec3));

    // set sbt data
    set_field(geom_data, "matId", e.materialId);
    set_field(geom_data, "lightId", e.lightId);
    set_field(geom_data, "vertex", vertex_buffer);
    set_field(geom_data, "normal", normal_buffer);
    set_field(geom_data, "index", index_buffer);

    od.geoms.push_back(geom_data);
}

template<typename T>
std::vector<T> to_vector(std::vector<std::tuple<std::string, T>> const* data)
{
    if (data == nullptr)
        return {};

    std::vector<T> result;
    result.reserve(data->size());

    for (auto const& [name, value] : *data)
        result.push_back(value);

    return result;
}

int32_t get_input(int32_t min = 0, int32_t max = 10)
{
    std::string tmp{};
    while (true)
    {
        fmt::print(" > ");
        std::getline(std::cin, tmp);
        if (tmp.empty()) continue;

        auto const s{ std::stoi(tmp) };
        if (s < min || max <= s) continue;

        return s;
    }
}


void render(camera_data const& camera,
            std::vector<std::tuple<std::string, material_data>> const *materials,
            std::vector<std::tuple<std::string, light_data>> const *lights)
{
    // 1) set mesh data into buffers
    if (od.geoms.empty())
    {
        od.world = create_instance_group(od.context, 0, nullptr);
        build_group_acceleration_structure(od.world);
    } else
    {
        // Create Geom group and build world
        auto triangles_group =
                owlTrianglesGeomGroupCreate(od.context, od.geoms.size(), od.geoms.data());
        build_group_acceleration_structure(triangles_group);

        // Create an Instance group to make world
        od.world = create_instance_group(od.context, 1, &triangles_group);
        build_group_acceleration_structure(od.world);
    }

    // 2) set miss program data
    /* INSERT HERE PROGRAM DATA */

    // 4) set ray gen data
    set_field(od.ray_gen_program, "fbPtr", od.frame_buffer);
    set_field(od.ray_gen_program, "fbSize", od.buffer_size);
    set_field(od.ray_gen_program, "camera.origin", camera.origin);
    set_field(od.ray_gen_program, "camera.llc", camera.llc);
    set_field(od.ray_gen_program, "camera.horizontal", camera.horizontal);
    set_field(od.ray_gen_program, "camera.vertical", camera.vertical);

    auto const material_list = to_vector(materials);
    auto const light_list = to_vector(lights);

    // 5) set launch params
    auto material_buffer{
            create_device_buffer(od.context, OWL_USER_TYPE(material_data), material_list.size(), material_list.data())
    };

    auto light_buffer{
            create_device_buffer(od.context, OWL_USER_TYPE(light_data), light_list.size(), light_list.data())
    };

    set_field(od.launch_params, "max_path_depth", reinterpret_cast<void*>(&od.max_path_depth));
    set_field(od.launch_params, "max_samples", reinterpret_cast<void*>(&od.max_samples));
    set_field(od.launch_params, "material_buffer", material_buffer);
    set_field(od.launch_params, "light_buffer", light_buffer);
    set_field(od.launch_params, "world", od.world);
    set_field(od.launch_params, "environment_map", od.environment_map);
    set_field(od.launch_params, "use_environment_map", od.use_environment_map);

    // 6) build sbt tables and load data
    build_optix(od.context);

    // 7) compute image
    fmt::print(fg(color::start), "LAUNCH TRACER\n");
    owlLaunch2D(od.ray_gen_program, od.buffer_size.x, od.buffer_size.y, od.launch_params);
    fmt::print(fg(color::stop), "STOP TRACER\n");
}

std::tuple<std::string, camera> select_scene(std::vector<std::tuple<std::string, camera>> const& scenes)
{
    auto counter{ 0 };
    for (auto const& [name, camera] : scenes)
        fmt::print(fg(color::start), "SCENE[{}]: {}\n", counter++, name);

    auto const s{ get_input(0, static_cast<int32_t>(scenes.size())) };
    return scenes[s];
}

std::vector<entity> load_scene(std::vector<std::tuple<std::string, std::shared_ptr<mesh>>> const& meshes,
                               std::vector<std::tuple<std::string, material_data>> const *materials,
                               std::vector<std::tuple<std::string, light_data>> const *lights)
{
    fmt::print(fg(color::log), "> MATERIALS\n");
    std::vector<entity> entities{};

    for (auto const& [name, mesh] : meshes)
    {
        entities.push_back(entity{
            .mesh_ptr = mesh.get(),
            .materialId = -1,
            .lightId = -1
        });
    }

    if (materials)
    {
        auto counter{ 0 };
        for (auto const& [name, material] : *materials)
            fmt::print(fg(color::log), "[{}] {}\n", counter++, name);

        counter = 0;
        for (auto const& [name, mesh] : meshes)
        {
            fmt::print("Object: {}\n", name);
            auto const s{ get_input(0, static_cast<int32_t>(materials->size())) };
            entities[counter++].materialId = s;
        }
    }

    if (lights)
    {
        auto counter{ 0 };
        for (auto const& [name, light] : *lights)
            fmt::print(fg(color::log), "[{}] {}\n", counter++, name);

        counter = 0;
        for (auto const& [name, mesh] : meshes)
        {
            fmt::print("Object: {}\n", name);
            auto const s{ get_input(0, static_cast<int32_t>(lights->size())) };
            entities[counter++].lightId = s;
        }
    }

    return entities;
}

int main(int argc, char **argv)
{
    auto prefix_path{ std::string{"."} };
    if (argc > 1) prefix_path = argv[1];
    auto const config_file{prefix_path + "/config.json"};

    auto const scenes{ parse_scenes(config_file) };
    auto const materials{ parse_materials(config_file) };
    auto const [scene_name, scene_camera] = select_scene(scenes);

    auto environment{ load_image("environment.hdr", prefix_path + "/assets/") };
    environment.ptr_tag = image_buffer::tag::allocated;

    auto const meshes{load_obj(fmt::format("{}/{}{}{}", prefix_path, "assets/", scene_name, ".obj.scene")) };
    auto const entities = load_scene(meshes, &materials, nullptr);

    optix_init();
    optix_set_environment_map(environment);

    od.buffer_size = ivec2{ 1024 };
    od.frame_buffer = create_pinned_host_buffer(od.context, OWL_INT, od.buffer_size.x * od.buffer_size.y);
    od.use_environment_map = true;
    od.max_samples = 128;
    od.max_path_depth = 64;


    for (auto e: entities) add(e);

    render(to_camera_data(scene_camera, od.buffer_size), &materials, nullptr);

    // copy image buffer

    image_buffer result{ od.buffer_size.x, od.buffer_size.y,
                        (uint32_t*)buffer_to_pointer(od.frame_buffer, 0), image_buffer::tag::referenced };
    write_image(result, fmt::format("{}.png", scene_name), prefix_path);

    optix_destroy();
}
