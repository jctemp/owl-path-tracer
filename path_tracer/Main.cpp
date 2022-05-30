
#include "utils/image_buffer.hpp"
#include "utils/mesh_loader.hpp"
#include "utils/parser.hpp"
#include "application.hpp"

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
    mesh *mesh_ptr;
    int32_t materialId{-1};
};

template <typename T>
std::vector<T> to_vector(std::vector<std::tuple<std::string, T>> const *data)
{
    if (data == nullptr)
        return {};

    std::vector<T> result;
    result.reserve(data->size());

    for (auto const &[name, value] : *data)
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
        if (tmp.empty())
            continue;

        auto const s{std::stoi(tmp)};
        if (s < min || max <= s)
            continue;

        return s;
    }
}

std::tuple<std::string, camera> select_scene(std::vector<std::tuple<std::string, camera>> const &scenes)
{
    auto counter{0};
    for (auto const &[name, camera] : scenes)
        fmt::print(fg(color::start), "SCENE[{}]: {}\n", counter++, name);

    auto const s{get_input(0, static_cast<int32_t>(scenes.size()))};
    return scenes[s];
}

int main(int argc, char **argv)
{
    std::string scene{"dragon"};
    auto const config_file{fmt::format("{}/assets/{}.json", std::filesystem::current_path().string(), scene)};

    auto const buffer_size = ivec2{1024};
    auto const max_samples = 128;
    auto const max_path_depth = 16;

    bool const environment_use = false;
    bool const environment_auto = false;
    vec3 const environment_color{vec3{1.0f, 1.0f, 1.0f}};
    float const environment_intensity{1.0f};

    /// PREPARE RENDERING

    auto const camera{parse_camera(config_file, buffer_size)};
    auto const materials{parse_materials(config_file)};
    auto const meshes{load_obj(fmt::format("{}/assets/{}.obj.scene", std::filesystem::current_path().string(), scene))};

    std::vector<entity> entities{};
    for (auto const &[name, mesh] : meshes)
    {
        int32_t position{0};
        for (auto const &[material_name, material] : materials)
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

    owl_data data{};
    init_data(data);

    std::vector<geom> geoms{};
    std::vector<buffer> indices_buffer_list{};
    std::vector<buffer> vertices_buffer_list{};
    std::vector<buffer> normals_buffer_list{};

    int32_t mesh_id{0};
    for (auto e : entities)
    {
        mesh &mesh{*e.mesh_ptr};

        auto &vertices{mesh.vertices};
        auto &indices{mesh.indices};
        auto &normals{mesh.normals};

        buffer vertex_buffer{create_device_buffer(data.owl_context, OWL_FLOAT3, vertices.size(), vertices.data())};
        buffer normal_buffer{create_device_buffer(data.owl_context, OWL_FLOAT3, normals.size(), normals.data())};
        buffer index_buffer{create_device_buffer(data.owl_context, OWL_INT3, indices.size(), indices.data())};

        indices_buffer_list.push_back(index_buffer);
        vertices_buffer_list.push_back(vertex_buffer);
        normals_buffer_list.push_back(normal_buffer);

        geom geom_data{owlGeomCreate(data.owl_context, data.triangle_geom)};
        set_triangle_vertices(geom_data, vertex_buffer, vertices.size(), sizeof(vec3));
        set_triangle_indices(geom_data, index_buffer, indices.size(), sizeof(ivec3));

        set_field(geom_data, "mesh_index", mesh_id++);
        set_field(geom_data, "material_index", e.materialId);

        geoms.push_back(geom_data);
    }

    init_world(data, geoms);

    auto const environment_map_texture{
        create_texture(data.owl_context, {environment_map.width, environment_map.height}, environment_map.buffer)};
    auto const frame_buffer{create_pinned_host_buffer(data.owl_context, OWL_INT, buffer_size.x * buffer_size.y)};

    auto vec_material = to_vector(&materials);

    auto material_buffer{create_device_buffer(data.owl_context, OWL_USER_TYPE(material_data),
                                              vec_material.size(), vec_material.data())};

    auto vertices_buffer{create_device_buffer(data.owl_context, OWL_BUFFER, vertices_buffer_list.size(),
                                              vertices_buffer_list.data())};
    auto indices_buffer{create_device_buffer(data.owl_context, OWL_BUFFER, indices_buffer_list.size(),
                                             indices_buffer_list.data())};
    auto normals_buffer{create_device_buffer(data.owl_context, OWL_BUFFER, normals_buffer_list.size(),
                                             normals_buffer_list.data())};

    set_field(data.ray_gen_prog, "fb_ptr", frame_buffer);
    set_field(data.ray_gen_prog, "fb_size", buffer_size);
    set_field(data.ray_gen_prog, "camera.origin", camera.origin);
    set_field(data.ray_gen_prog, "camera.llc", camera.llc);
    set_field(data.ray_gen_prog, "camera.horizontal", camera.horizontal);
    set_field(data.ray_gen_prog, "camera.vertical", camera.vertical);

    set_field(data.lp, "max_path_depth", max_path_depth);
    set_field(data.lp, "max_samples", max_samples);
    set_field(data.lp, "material_buffer", material_buffer);
    set_field(data.lp, "vertices_buffer", vertices_buffer);
    set_field(data.lp, "indices_buffer", indices_buffer);
    set_field(data.lp, "normals_buffer", normals_buffer);
    set_field(data.lp, "world", data.world);
    set_field(data.lp, "environment_map", environment_map_texture);
    set_field(data.lp, "environment_use", environment_use);
    set_field(data.lp, "environment_auto", environment_auto);
    set_field(data.lp, "environment_color", environment_color);
    set_field(data.lp, "environment_intensity", environment_intensity);

    owlBuildPrograms(data.owl_context);
    owlBuildPipeline(data.owl_context);
    owlBuildSBT(data.owl_context);

    std::vector<vec3> greys{
        {1.0f},
        {0.9f},
        {0.8f},
        {0.7f},
        {0.6f},
        {0.5f},
        {0.4f},
        {0.3f},
        {0.2f},
        {0.1f},
        {0.0f},
    };

    for (auto const &grey : greys)
    {
        vec_material[0].base_color = grey;

        owlBufferRelease(material_buffer);
        material_buffer = create_device_buffer(data.owl_context, OWL_USER_TYPE(material_data),
                                              vec_material.size(), vec_material.data());
        set_field(data.lp, "material_buffer", material_buffer);

        fmt::print(fg(color::start), "TRACING\n");
        owlLaunch2D(data.ray_gen_prog, buffer_size.x, buffer_size.y, data.lp);
        image_buffer result{buffer_size.x, buffer_size.y,
                            reinterpret_cast<uint32_t const *>(buffer_to_pointer(frame_buffer, 0)),
                            image_buffer::tag::referenced};
        write_image(result, fmt::format("{}-{}.png", grey.x, scene), std::filesystem::current_path().string());

    }

    destroy_context(data.owl_context);
}
