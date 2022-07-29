
#include "utils/image_buffer.hpp"
#include "utils/mesh_loader.hpp"
#include "utils/parser.hpp"

#include "application.hpp"

#include <fmt/core.h>
#include <fmt/color.h>
#include <filesystem>
#include <stb_image.h>

extern "C" char device_ptx[];

template <typename T>
std::vector<T> to_vector(std::vector<std::tuple<std::string, T, std::string>> const *data)
{
    if (data == nullptr)
        return {};

    std::vector<T> result;
    result.reserve(data->size());

    for (auto const &[name, value, _] : *data)
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

/// Load owl data to prepare optix rendering
void init_owl_data(owl_data& data)
{
    /// create context and module to prepare different components
    data.owl_context = create_context(nullptr, 1);
    data.owl_module = create_module(data.owl_context, device_ptx);


    /// create bindable data for triangles
    var_decl triangles_geom_vars
            {
                    {"mesh_index",     OWL_INT,     OWL_OFFSETOF(entity_data, mesh_index)},
                    {"material_index", OWL_INT,     OWL_OFFSETOF(entity_data, material_index)},
                    {"has_texture",    OWL_BOOL,    OWL_OFFSETOF(entity_data, has_texture)},
                    {"texture",        OWL_TEXTURE, OWL_OFFSETOF(entity_data, texture)},
                    {nullptr}
            };

    data.triangle_geom = create_geom_type(data.owl_context, OWL_GEOM_TRIANGLES,
            sizeof(entity_data), triangles_geom_vars);

    geom_type_closest_hit_program(data.triangle_geom, data.owl_module, "triangle_hit", 0);


    /// create programs for the pipeline
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

    data.ray_gen_prog = create_ray_gen_program(data.owl_context, data.owl_module,
            "ray_gen", sizeof(ray_gen_data), ray_gen_vars);


    data.miss_prog = create_miss_program(data.owl_context, data.owl_module, "miss", 0u, nullptr);
    data.miss_shadow_prog = create_miss_program(data.owl_context, data.owl_module, "miss_shadow", 0u, nullptr);
    owlMissProgSet(data.owl_context, 0, data.miss_prog);
    owlMissProgSet(data.owl_context, 1, data.miss_shadow_prog);


    /// create launch parameters
    var_decl launchParamsVars
            {
                    {"max_path_depth",        OWL_INT,     OWL_OFFSETOF(launch_params_data, max_path_depth)},
                    {"max_samples",           OWL_INT,     OWL_OFFSETOF(launch_params_data, max_samples)},
                    {"material_buffer",       OWL_BUFFER,  OWL_OFFSETOF(launch_params_data, material_buffer)},
                    {"vertices_buffer",       OWL_BUFFER,  OWL_OFFSETOF(launch_params_data, vertices_buffer)},
                    {"indices_buffer",        OWL_BUFFER,  OWL_OFFSETOF(launch_params_data, indices_buffer)},
                    {"normals_buffer",        OWL_BUFFER,  OWL_OFFSETOF(launch_params_data, normals_buffer)},
                    {"texcoords_buffer",      OWL_BUFFER,  OWL_OFFSETOF(launch_params_data, texcoords_buffer)},
                    {"world",                 OWL_GROUP,   OWL_OFFSETOF(launch_params_data, world)},
                    {"environment_map",       OWL_TEXTURE, OWL_OFFSETOF(launch_params_data, environment_map)},
                    {"environment_use",       OWL_BOOL,    OWL_OFFSETOF(launch_params_data, environment_use)},
                    {"environment_auto",      OWL_BOOL,    OWL_OFFSETOF(launch_params_data, environment_auto)},
                    {"environment_color",     OWL_FLOAT3,  OWL_OFFSETOF(launch_params_data, environment_color)},
                    {"environment_intensity", OWL_FLOAT,   OWL_OFFSETOF(launch_params_data, environment_intensity)},
                    {nullptr}
            };

    data.lp = create_launch_params(data.owl_context, sizeof(launch_params_data), launchParamsVars);


    /// necessary for the building geometries
    owlBuildPrograms(data.owl_context); 
}

/// create IAS acceleration structure based on geoms
void init_owl_world(owl_data& data, std::vector<geom>& geoms)
{
    if (geoms.empty()) throw std::runtime_error("no geometries");

    auto triangles_group{owlTrianglesGeomGroupCreate(data.owl_context, geoms.size(), geoms.data())};
    build_group_acceleration_structure(triangles_group);

    data.world = create_instance_group(data.owl_context, 1, &triangles_group);
    build_group_acceleration_structure(data.world);
}

/// Load program data from settings file in assets folder
void init_program_data(program_data& pdata, test_data& tdata, std::string const& assets_path)
{
    auto const settings{parse_settings(fmt::format("{}/{}", assets_path, "settings.json"))};

    auto const config_file{ fmt::format("{}/{}.json", assets_path, settings.scene) };

    tdata = settings.test;

    pdata.scene = settings.scene;
    pdata.buffer_size = settings.buffer_size;
    pdata.max_path_depth = settings.max_path_depth;
    pdata.max_samples = settings.max_samples;
    pdata.environment_use = settings.environment_use;
    pdata.environment_auto = settings.environment_auto;
    pdata.environment_color = settings.environment_color;
    pdata.environment_intensity = settings.environment_intensity;

    pdata.environment_map = load_image("environment.hdr", assets_path);

    pdata.camera = parse_camera(config_file, pdata.buffer_size);
    pdata.materials = parse_materials(config_file);
    pdata.meshes = load_obj(fmt::format("{}/{}.obj.scene", assets_path, settings.scene));

    pdata.entities = std::vector<entity>{};
    for (auto const &[name, mesh] : pdata.meshes)
    {
        int32_t position{0};
        for (auto const &[material_name, material, _] : pdata.materials)
        {
            if (material_name == name)
            {
                pdata.entities.push_back({.mesh_ptr = mesh.get(), .materialId = position});
                break;
            }
            ++position;
        }
    }

}

/// initial create data and bind it to the sbt
void bind_sbt_data(program_data& pdata, owl_data& data, std::string const& assets_path)
{
    /// Create sbt data for renderer
    int32_t mesh_id{0};
    for (auto e : pdata.entities)
    {
        mesh &mesh{*e.mesh_ptr};

        auto &vertices{mesh.vertices};
        auto &indices{mesh.indices};
        auto &normals{mesh.normals};
        auto& texcoods{ mesh.texcoords };

        buffer vertex_buffer = create_device_buffer(data.owl_context, OWL_FLOAT3, vertices.size(), vertices.data());
        buffer normal_buffer = create_device_buffer(data.owl_context, OWL_FLOAT3, normals.size(), normals.data());
        buffer index_buffer = create_device_buffer(data.owl_context, OWL_INT3, indices.size(), indices.data());
        buffer texcoords_buffer = create_device_buffer(data.owl_context, OWL_FLOAT2, texcoods.size(), texcoods.data());

        pdata.indices_buffer_list.push_back(index_buffer);
        pdata.vertices_buffer_list.push_back(vertex_buffer);
        pdata.normals_buffer_list.push_back(normal_buffer);
        pdata.texcoords_buffer_list.push_back(texcoords_buffer);

        geom geom_data{owlGeomCreate(data.owl_context, data.triangle_geom)};
        set_triangle_vertices(geom_data, vertex_buffer, vertices.size(), sizeof(vec3));
        set_triangle_indices(geom_data, index_buffer, indices.size(), sizeof(ivec3));

        set_field(geom_data, "mesh_index", mesh_id++);
        set_field(geom_data, "material_index", e.materialId);
        
        auto&[tmp0, tmp1, filename] = pdata.materials[e.materialId];
        if (!filename.empty())
        {
            auto file_path = assets_path + "/" + filename;

            if (!std::filesystem::exists(file_path))
            {
                fmt::print(fg(color::warn), "Image file {} does not exist. Continue with empty.\n", file_path);
                return;
            }

            int32_t width, height, comp;
            auto buffer{ reinterpret_cast<uint32_t*>(stbi_load(file_path.c_str(), &width,
                    &height, &comp, STBI_rgb_alpha)) };

            for (int32_t y{ 0 }; y < height / 2; y++)
            {
                uint32_t* line_y{ buffer + y * width };
                uint32_t* mirrored_y{ buffer + (height - 1 - y) * width };
                for (int x = 0; x < width; x++) std::swap(line_y[x], mirrored_y[x]);
            }

            texture tex = owlTexture2DCreate(data.owl_context,
                OWL_TEXEL_FORMAT_RGBA8,
                width, height, buffer,
                OWL_TEXTURE_NEAREST,
                OWL_TEXTURE_CLAMP);

            set_field(geom_data, "texture", tex);
            set_field(geom_data, "has_texture", true);

            delete[] buffer;
        }

        pdata.geoms.push_back(geom_data);
    }

    init_owl_world(data, pdata.geoms);

    auto const environment_map_texture{
            create_texture(data.owl_context, {pdata.environment_map.width, pdata.environment_map.height}, pdata.environment_map.buffer)};
    pdata.framebuffer = create_pinned_host_buffer(data.owl_context, OWL_INT, pdata.buffer_size.x * pdata.buffer_size.y);

    auto vec_material = to_vector(&pdata.materials);
    pdata.material_buffer = {create_device_buffer(data.owl_context, OWL_USER_TYPE(material_data),
            vec_material.size(), vec_material.data())};
    pdata.vertices_buffer = {create_device_buffer(data.owl_context, OWL_BUFFER, pdata.vertices_buffer_list.size(),
            pdata.vertices_buffer_list.data())};
    pdata.indices_buffer = {create_device_buffer(data.owl_context, OWL_BUFFER, pdata.indices_buffer_list.size(),
            pdata.indices_buffer_list.data())};
    pdata.normals_buffer = {create_device_buffer(data.owl_context, OWL_BUFFER, pdata.normals_buffer_list.size(),
            pdata.normals_buffer_list.data())};
    pdata.texcoords_buffer = { create_device_buffer(data.owl_context, OWL_BUFFER, pdata.texcoords_buffer_list.size(),
        pdata.texcoords_buffer_list.data()) };

    /// bind sbt data
    set_field(data.ray_gen_prog, "fb_ptr", pdata.framebuffer);
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
    set_field(data.lp, "texcoords_buffer", pdata.texcoords_buffer);
    set_field(data.lp, "world", data.world);
    set_field(data.lp, "environment_map", environment_map_texture);
    set_field(data.lp, "environment_use", pdata.environment_use);
    set_field(data.lp, "environment_auto", pdata.environment_auto);
    set_field(data.lp, "environment_color", pdata.environment_color);
    set_field(data.lp, "environment_intensity", pdata.environment_intensity);

    owlBuildPrograms(data.owl_context);
    owlBuildPipeline(data.owl_context);
    owlBuildSBT(data.owl_context);
}

/// re-binds material buffer
void reset_field(owl_data& odata, program_data& pdata)
{
    owlBufferRelease(pdata.material_buffer);
    auto vec_material = to_vector(&pdata.materials);
    pdata.material_buffer = create_device_buffer(odata.owl_context, OWL_USER_TYPE(material_data),
            vec_material.size(), vec_material.data());
    set_field(odata.lp, "material_buffer", pdata.material_buffer);
}

/// finds material in material map
auto get_material(std::vector<std::tuple<std::string, material_data, std::string>>& materials, const test_data& test)
{// get material with name
    auto material_tuple{
            std::find_if(std::begin(materials), std::end(materials), [test] (std::tuple<std::string, material_data, std::string> const &t)
            {
                return std::get<0>(t) == test.material_name;
            })
    };
    material_data *material{&std::get<1>(*material_tuple)};
    return material;
}

/// modifies material data
void modify_sbt(owl_data &odata, program_data &pdata, std::vector<std::tuple<std::string, material_data, std::string>> &materials,
                test_data const &test, vec3 value)
{
    auto material = get_material(materials, test);
    material->base_color = value;
    reset_field(odata, pdata);
}

/// modifies material data
void modify_sbt(owl_data &odata, program_data &pdata, std::vector<std::tuple<std::string, material_data, std::string>> &materials,
                test_data const &test, float value)
{
    auto material = get_material(materials, test);
    if (test.attribute_name == "subsurface")
        material->subsurface = value;
    else if (test.attribute_name == "metallic")
        material->metallic = value;
    else if (test.attribute_name == "specular")
        material->specular = value;
    else if (test.attribute_name == "specular_tint")
        material->specular_tint = value;
    else if (test.attribute_name == "roughness")
        material->roughness = value;
    else if (test.attribute_name == "anisotropic")
        material->anisotropic = value;
    else if (test.attribute_name == "sheen")
        material->sheen = value;
    else if (test.attribute_name == "sheen_tint")
        material->sheen_tint = value;
    else if (test.attribute_name == "clearcoat")
        material->clearcoat = value;
    else if (test.attribute_name == "clearcoat_gloss")
        material->clearcoat_gloss = value;
    else if (test.attribute_name == "ior")
        material->ior = value;
    else if (test.attribute_name == "specular_transmission")
        material->specular_transmission = value;
    else if (test.attribute_name == "specular_transmission_roughness")
        material->specular_transmission_roughness = value;
    reset_field(odata, pdata);
}

/// renders the scene
void render_frame(owl_data& data, program_data& pdata, test_data& tdata, std::string const& values)
{
    fmt::print(fg(color::start), "TRACING\n");
    owlLaunch2D(data.ray_gen_prog, pdata.buffer_size.x, pdata.buffer_size.y, data.lp);
    image_buffer result{pdata.buffer_size.x, pdata.buffer_size.y,
                        reinterpret_cast<uint32_t const *>(buffer_to_pointer(pdata.framebuffer, 0)),
                        image_buffer::tag::referenced};
    write_image(result, fmt::format("{}_{}_{}({}).png", pdata.scene, tdata.name, tdata.attribute_name, values), std::filesystem::current_path().string());
}