
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
};

template<typename T>
std::vector<T> to_vector(std::vector<std::tuple<std::string, T>> const* data)
{
    if (data == nullptr)
        return {};

    std::vector<T> result;
    result.reserve(data->size());

    for (auto const& [name, value]: *data)
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

        auto const s{std::stoi(tmp)};
        if (s < min || max <= s) continue;

        return s;
    }
}

std::tuple<std::string, camera> select_scene(std::vector<std::tuple<std::string, camera>> const& scenes)
{
    auto counter{0};
    for (auto const& [name, camera]: scenes)
        fmt::print(fg(color::start), "SCENE[{}]: {}\n", counter++, name);

    auto const s{get_input(0, static_cast<int32_t>(scenes.size()))};
    return scenes[s];
}

int main(int argc, char** argv)
{
    std::string scene{"dragon"};
    auto const config_file{fmt::format("{}/assets/{}.json", std::filesystem::current_path().string(), scene)};

    auto const buffer_size = ivec2{1024};
    auto const max_samples = 1024;
    auto const max_path_depth = 64;

    bool const environment_use = false;
    bool const environment_auto = false;
    vec3 const environment_color{vec3{1.0f, 0.0f, 1.0f}};
    float const environment_intensity{1.0f};

    /// PREPARE RENDERING

    auto const camera{parse_camera(config_file, buffer_size)};
    auto const materials{parse_materials(config_file)};
    auto const meshes{load_obj(fmt::format("{}/assets/{}.obj.scene", std::filesystem::current_path().string(), scene))};

    std::vector<entity> entities{};
    for (auto const& [name, mesh]: meshes)
    {
        int32_t position{0};
        for (auto const& [material_name, material]: materials)
        {
            if (material_name == name)
            {
                entities.push_back({.mesh_ptr = mesh.get(), .materialId = position});
                break;
            }
            ++position;
        }
    }

    image_buffer environment_map{load_image("environment.hdr", std::filesystem::current_path().string() + "/assets/")};
    environment_map.ptr_tag = image_buffer::tag::allocated;

#pragma region "OWL INIT"

    OWLContext owl_context = create_context(nullptr, 1);
    OWLModule owl_module = create_module(owl_context, device_ptx);

#pragma endregion

#pragma region "CREATE GEOMETRY TYPES"

    var_decl triangles_geom_vars
            {
                    {"mesh_index",     OWL_INT,    OWL_OFFSETOF(entity_data, mesh_index)},
                    {"material_index", OWL_INT,    OWL_OFFSETOF(entity_data, material_index)},
                    {nullptr}
            };

    geom_type triangle_geom{create_geom_type(owl_context, OWL_GEOM_TRIANGLES,
            sizeof(entity_data), triangles_geom_vars)};

    geom_type_closest_hit_program(triangle_geom, owl_module, "triangle_hit", 0);

    owlBuildPrograms(owl_context); // necessary for the building geometries

#pragma endregion

#pragma region "SET GEOMETRY DATA"

    std::vector<geom> geoms{};
    std::vector<buffer> indices_buffer_list{};
    std::vector<buffer> vertices_buffer_list{};
    std::vector<buffer> normals_buffer_list{};

    int32_t mesh_id{0};
    for (auto e: entities)
    {
        mesh& mesh{*e.mesh_ptr};

        auto& vertices{mesh.vertices};
        auto& indices{mesh.indices};
        auto& normals{mesh.normals};

        buffer vertex_buffer{create_device_buffer(owl_context, OWL_FLOAT3, vertices.size(), vertices.data())};
        buffer normal_buffer{create_device_buffer(owl_context, OWL_FLOAT3, normals.size(), normals.data())};
        buffer index_buffer{create_device_buffer(owl_context, OWL_INT3, indices.size(), indices.data())};

        indices_buffer_list.push_back(index_buffer);
        vertices_buffer_list.push_back(vertex_buffer);
        normals_buffer_list.push_back(normal_buffer);

        geom geom_data{owlGeomCreate(owl_context, triangle_geom)};
        set_triangle_vertices(geom_data, vertex_buffer, vertices.size(), sizeof(vec3));
        set_triangle_indices(geom_data, index_buffer, indices.size(), sizeof(ivec3));

        set_field(geom_data, "mesh_index", mesh_id++);
        set_field(geom_data, "material_index", e.materialId);

        geoms.push_back(geom_data);
    }

#pragma endregion

#pragma region "CREATE IAS FROM GEOMETRY"

    if (geoms.empty()) throw std::runtime_error("no geometries");

    auto triangles_group{owlTrianglesGeomGroupCreate(owl_context, geoms.size(), geoms.data())};
    build_group_acceleration_structure(triangles_group);

    auto world{create_instance_group(owl_context, 1, &triangles_group)};
    build_group_acceleration_structure(world);

#pragma endregion

#pragma region "SET MISS AND RAY GEN PROGRAM"

    var_decl ray_gen_vars
            {
                    {"fb_ptr",            OWL_BUFPTR, OWL_OFFSETOF(ray_gen_data, fb_ptr)},
                    {"fb_size",           OWL_INT2,   OWL_OFFSETOF(ray_gen_data, fb_size)},
                    {"camera.origin",     OWL_FLOAT3, OWL_OFFSETOF(ray_gen_data, camera.origin)},
                    {"camera.llc",        OWL_FLOAT3, OWL_OFFSETOF(ray_gen_data, camera.llc)},
                    {"camera.horizontal", OWL_FLOAT3, OWL_OFFSETOF(ray_gen_data, camera.horizontal)},
                    {"camera.vertical",   OWL_FLOAT3, OWL_OFFSETOF(ray_gen_data, camera.vertical)},
                    {nullptr}
            };

    ray_gen_program ray_gen_prog{create_ray_gen_program(owl_context, owl_module,
            "ray_gen", sizeof(ray_gen_data), ray_gen_vars)};


    miss_program miss_prog{create_miss_program(owl_context, owl_module, "miss", 0u, nullptr)};
    miss_program miss_shadow_prog = create_miss_program(owl_context, owl_module, "miss_shadow", 0u, nullptr);
    owlMissProgSet(owl_context, 0, miss_prog);
    owlMissProgSet(owl_context, 1, miss_shadow_prog);

#pragma endregion

#pragma region "SET LAUNCH PARAMS"

    var_decl launchParamsVars
            {
                    {"max_path_depth",        OWL_INT,     OWL_OFFSETOF(launch_params_data, max_path_depth)},
                    {"max_samples",           OWL_INT,     OWL_OFFSETOF(launch_params_data, max_samples)},
                    {"material_buffer",       OWL_BUFFER,  OWL_OFFSETOF(launch_params_data, material_buffer)},
                    {"vertices_buffer",       OWL_BUFFER,  OWL_OFFSETOF(launch_params_data, vertices_buffer)},
                    {"indices_buffer",        OWL_BUFFER,  OWL_OFFSETOF(launch_params_data, indices_buffer)},
                    {"normals_buffer",        OWL_BUFFER,  OWL_OFFSETOF(launch_params_data, normals_buffer)},
                    {"world",                 OWL_GROUP,   OWL_OFFSETOF(launch_params_data, world)},
                    {"environment_map",       OWL_TEXTURE, OWL_OFFSETOF(launch_params_data, environment_map)},
                    {"environment_use",       OWL_BOOL,    OWL_OFFSETOF(launch_params_data, environment_use)},
                    {"environment_auto",      OWL_BOOL,    OWL_OFFSETOF(launch_params_data, environment_auto)},
                    {"environment_color",     OWL_FLOAT3,  OWL_OFFSETOF(launch_params_data, environment_color)},
                    {"environment_intensity", OWL_FLOAT,   OWL_OFFSETOF(launch_params_data, environment_intensity)},
                    {nullptr}
            };

    launch_params lp{create_launch_params(owl_context, sizeof(launch_params_data), launchParamsVars)};

#pragma endregion

#pragma region "SET VARIABLES"

    auto const environment_map_texture{
            create_texture(owl_context, {environment_map.width, environment_map.height}, environment_map.buffer)};
    auto const frame_buffer{create_pinned_host_buffer(owl_context, OWL_INT, buffer_size.x * buffer_size.y)};

    auto const vec_material = to_vector(&materials);

    auto material_buffer{create_device_buffer(owl_context, OWL_USER_TYPE(material_data),
            vec_material.size(), vec_material.data())};

    auto vertices_buffer{create_device_buffer(owl_context, OWL_BUFFER, vertices_buffer_list.size(),
            vertices_buffer_list.data())};
    auto indices_buffer{create_device_buffer(owl_context, OWL_BUFFER, indices_buffer_list.size(),
            indices_buffer_list.data())};
    auto normals_buffer{create_device_buffer(owl_context, OWL_BUFFER, normals_buffer_list.size(),
            normals_buffer_list.data())};

    set_field(ray_gen_prog, "fb_ptr", frame_buffer);
    set_field(ray_gen_prog, "fb_size", buffer_size);
    set_field(ray_gen_prog, "camera.origin", camera.origin);
    set_field(ray_gen_prog, "camera.llc", camera.llc);
    set_field(ray_gen_prog, "camera.horizontal", camera.horizontal);
    set_field(ray_gen_prog, "camera.vertical", camera.vertical);

    set_field(lp, "max_path_depth", max_path_depth);
    set_field(lp, "max_samples", max_samples);
    set_field(lp, "material_buffer", material_buffer);
    set_field(lp, "vertices_buffer", vertices_buffer);
    set_field(lp, "indices_buffer", indices_buffer);
    set_field(lp, "normals_buffer", normals_buffer);
    set_field(lp, "world", world);
    set_field(lp, "environment_map", environment_map_texture);
    set_field(lp, "environment_use", environment_use);
    set_field(lp, "environment_auto", environment_auto);
    set_field(lp, "environment_color", environment_color);
    set_field(lp, "environment_intensity", environment_intensity);

    owlBuildPrograms(owl_context);
    owlBuildPipeline(owl_context);
    owlBuildSBT(owl_context);

#pragma endregion

#pragma region "LAUNCH RAY TRACING API"

    // TODO: make materials updateable and re-launch ray tracing

    fmt::print(fg(color::start), "LAUNCH TRACER\n");
    owlLaunch2D(ray_gen_prog, buffer_size.x, buffer_size.y, lp);
    fmt::print(fg(color::stop), "STOP TRACER\n");

#pragma endregion

#pragma region "WRITING HOST PINNED BUFFER TO FILE"

    image_buffer result{buffer_size.x, buffer_size.y,
                        reinterpret_cast<uint32_t const*>(buffer_to_pointer(frame_buffer, 0)),
                        image_buffer::tag::referenced};
    write_image(result, fmt::format("{}.png", scene), std::filesystem::current_path().string());

#pragma endregion

    destroy_context(owl_context);
}
