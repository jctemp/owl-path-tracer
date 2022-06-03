
#include "utils/image_buffer.hpp"
#include "utils/mesh_loader.hpp"
#include "utils/parser.hpp"
#include "application.hpp"

#include "owl.hpp"
#include "camera.hpp"
#include "device/device_global.hpp"

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ranges.h>

#include <filesystem>

using namespace owl;

extern "C" char device_ptx[];



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

std::vector<float> to_vector(vec3 const &data)
{
    std::vector<float> result{};
    result.reserve(3);

    result.push_back(data.x);
    result.push_back(data.y);
    result.push_back(data.z);

    return result;
}

int main(int argc, char **argv)
{
    /// Load program data from settings file in assets folder
    program_data pdata{};
    init_program_data(pdata, std::filesystem::current_path().string() + "/assets");

    /// Load owl data to prepare optix rendering
    owl_data data{};
    init_owl_data(data);

    /// Create sbt data for renderer
    int32_t mesh_id{0};
    for (auto e : pdata.entities)
    {
        mesh &mesh{*e.mesh_ptr};

        auto &vertices{mesh.vertices};
        auto &indices{mesh.indices};
        auto &normals{mesh.normals};

        buffer vertex_buffer = create_device_buffer(data.owl_context, OWL_FLOAT3, vertices.size(), vertices.data());
        buffer normal_buffer = create_device_buffer(data.owl_context, OWL_FLOAT3, normals.size(), normals.data());
        buffer index_buffer = create_device_buffer(data.owl_context, OWL_INT3, indices.size(), indices.data());

        pdata.indices_buffer_list.push_back(index_buffer);
        pdata.vertices_buffer_list.push_back(vertex_buffer);
        pdata.normals_buffer_list.push_back(normal_buffer);

        geom geom_data{owlGeomCreate(data.owl_context, data.triangle_geom)};
        set_triangle_vertices(geom_data, vertex_buffer, vertices.size(), sizeof(vec3));
        set_triangle_indices(geom_data, index_buffer, indices.size(), sizeof(ivec3));

        set_field(geom_data, "mesh_index", mesh_id++);
        set_field(geom_data, "material_index", e.materialId);

        pdata.geoms.push_back(geom_data);
    }

    init_owl_world(data, pdata.geoms);

    auto const environment_map_texture{
        create_texture(data.owl_context, {pdata.environment_map.width, pdata.environment_map.height}, pdata.environment_map.buffer)};
    auto const frame_buffer{create_pinned_host_buffer(data.owl_context, OWL_INT, pdata.buffer_size.x * pdata.buffer_size.y)};

    auto vec_material = to_vector(&pdata.materials);

    pdata.material_buffer = {create_device_buffer(data.owl_context, OWL_USER_TYPE(material_data),
                                              vec_material.size(), vec_material.data())};
    pdata.vertices_buffer = {create_device_buffer(data.owl_context, OWL_BUFFER, pdata.vertices_buffer_list.size(),
                                              pdata.vertices_buffer_list.data())};
    pdata.indices_buffer = {create_device_buffer(data.owl_context, OWL_BUFFER, pdata.indices_buffer_list.size(),
                                             pdata.indices_buffer_list.data())};
    pdata.normals_buffer = {create_device_buffer(data.owl_context, OWL_BUFFER,pdata.normals_buffer_list.size(),
                                             pdata.normals_buffer_list.data())};

    /// bind sbt data
    set_field(data.ray_gen_prog, "fb_ptr", frame_buffer);
    set_field(data.ray_gen_prog, "fb_size", pdata.buffer_size);
    set_field(data.ray_gen_prog, "camera.origin", pdata.camera.origin);
    set_field(data.ray_gen_prog, "camera.llc", pdata.camera.llc);
    set_field(data.ray_gen_prog, "camera.horizontal", pdata.camera.horizontal);
    set_field(data.ray_gen_prog, "camera.vertical", pdata.camera.vertical);

    set_field(data.lp, "max_path_depth", pdata.max_path_depth);
    set_field(data.lp, "max_samples", pdata.max_samples);
    set_field(data.lp, "material_buffer", pdata.material_buffer);
    set_field(data.lp, "vertices_buffer", pdata.vertices_buffer);
    set_field(data.lp, "indices_buffer", pdata.indices_buffer);
    set_field(data.lp, "normals_buffer", pdata.normals_buffer);
    set_field(data.lp, "world", data.world);
    set_field(data.lp, "environment_map", environment_map_texture);
    set_field(data.lp, "environment_use", pdata.environment_use);
    set_field(data.lp, "environment_auto", pdata.environment_auto);
    set_field(data.lp, "environment_color", pdata.environment_color);
    set_field(data.lp, "environment_intensity", pdata.environment_intensity);

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

        owlBufferRelease(pdata.material_buffer);
        pdata.material_buffer = create_device_buffer(data.owl_context, OWL_USER_TYPE(material_data),
                                              vec_material.size(), vec_material.data());
        set_field(data.lp, "material_buffer", pdata.material_buffer);

        fmt::print(fg(color::start), "TRACING\n");
        owlLaunch2D(data.ray_gen_prog, pdata.buffer_size.x, pdata.buffer_size.y, data.lp);
        image_buffer result{pdata.buffer_size.x, pdata.buffer_size.y,
                            reinterpret_cast<uint32_t const *>(buffer_to_pointer(frame_buffer, 0)),
                            image_buffer::tag::referenced};
        write_image(result, fmt::format("{}-{}({:.1f}).png", pdata.scene, pdata.test_name, fmt::join(to_vector(grey), ",")), std::filesystem::current_path().string());

    }

    destroy_context(data.owl_context);
}
