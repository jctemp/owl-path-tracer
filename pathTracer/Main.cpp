
#include "utils/image_buffer.hpp"
#include "utils/mesh_loader.hpp"
#include "owl.hpp"
#include "camera.hpp"
#include "device/device_global.hpp"

#include <fmt/core.h>
#include <fmt/color.h>

using namespace owl;

extern "C" char device_ptx[];

struct entity
{
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
    od.context = create_context(nullptr, 1);
    od.module = create_module(od.context, device_ptx);
}

void optix_ray_gen_program()
{
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

    od.ray_gen_program =
            create_ray_gen_program(od.context, od.module, "ray_gen", sizeof(ray_gen_data), ray_gen_vars);
}

void optix_miss_program()
{

    var_decl miss_program_vars
            {
                    {nullptr}
            };

    od.miss_program =
            create_miss_program(od.context, od.module, "miss", sizeof(miss_data), miss_program_vars);
}

void optix_launch_params()
{
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

    od.launch_params =
            create_launch_params(od.context, sizeof(launch_params_data), launchParamsVars);
}

void optix_triangle_geom()
{
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
    if (od.environment_map != nullptr)
        destroy_texture(od.environment_map);

    od.environment_map = create_texture(
            od.context,
            {texture.width, texture.height},
            texture.buffer
    );
}

void add(mesh* m, entity e)
{
    mesh& mesh{*m};

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

void
render(camera_data const& camera, std::vector<material_data> const& materials, std::vector<light_data> const& lights)
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

    // 5) set launch params
    auto material_buffer{
            create_device_buffer(od.context, OWL_USER_TYPE(material_data), materials.size(), materials.data())
    };

    auto light_buffer{
            create_device_buffer(od.context, OWL_USER_TYPE(light_data), lights.size(), lights.data())
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

int main()
{
    auto const prefix_path{ std::string{"../.."} };
    // auto const prefix_path{std::string{"../../../.."}};

#pragma region data

    /* BSDF */
    material_data diffuse_dark{material_data::type::disney};
    diffuse_dark.base_color = vec3{0.2f};
    diffuse_dark.roughness = 0.0f;
    diffuse_dark.subsurface = 0.0f;
    diffuse_dark.metallic = 0.0f;
    diffuse_dark.specular = 0.0f;
    diffuse_dark.roughness = 0.5f;
    diffuse_dark.sheen = 0.5f;
    diffuse_dark.clearcoat = 0.0f;
    diffuse_dark.ior = 1.45f;

    material_data diffuse_bright{material_data::type::disney};
    diffuse_bright.base_color = vec3{0.8f};
    diffuse_bright.roughness = 0.0f;
    diffuse_bright.subsurface = 0.0f;
    diffuse_bright.metallic = 0.0f;
    diffuse_bright.specular = 0.0f;
    diffuse_bright.roughness = 0.5f;
    diffuse_bright.sheen = 0.5f;
    diffuse_bright.clearcoat = 0.0f;
    diffuse_bright.ior = 1.45f;

    auto const base_color_example{vec3{0.8f}};

    material_data disney_diffuse{material_data::type::disney_diffuse};
    disney_diffuse.base_color = base_color_example;
    disney_diffuse.ior = 1.45f;

    material_data disney_subsurface{material_data::type::disney_subsurface};
    disney_subsurface.subsurface_color = base_color_example;
    disney_subsurface.ior = 1.45f;

    material_data diseny_retro{material_data::type::disney_retro};
    diseny_retro.base_color = base_color_example;
    diseny_retro.roughness = 0.5f;
    diseny_retro.ior = 1.45f;

    material_data diseny_sheen{material_data::type::disney_sheen};
    diseny_sheen.base_color = base_color_example;
    diseny_sheen.ior = 1.45f;
    diseny_sheen.sheen = 0.3f;

    material_data diseny_clearcoat_glossy{material_data::type::disney_clearcoat};
    diseny_clearcoat_glossy.clearcoat = 1.0f;
    diseny_clearcoat_glossy.clearcoat_gloss = 1.0f;

    material_data diseny_clearcoat_rough{material_data::type::disney_clearcoat};
    diseny_clearcoat_rough.clearcoat = 1.0f;
    diseny_clearcoat_rough.clearcoat_gloss = 0.0f;

    material_data diseny_microfacet_glossy{material_data::type::disney_microfacet};
    diseny_microfacet_glossy.base_color = base_color_example;
    diseny_microfacet_glossy.roughness = 0.0f;
    diseny_microfacet_glossy.ior = 1.45f;

    material_data diseny_microfacet_rough{material_data::type::disney_microfacet};
    diseny_microfacet_rough.base_color = base_color_example;
    diseny_microfacet_rough.roughness = 1.0f;
    diseny_microfacet_rough.ior = 1.45f;

    material_data diffuse_brdf{material_data::type::disney};
    diffuse_bright.base_color = vec3{0.8f};
    diffuse_bright.roughness = 0.5f;
    diffuse_bright.metallic = 0.5f;
    diffuse_bright.specular = 0.5f;
    diffuse_bright.sheen = 0.5f;
    diffuse_bright.clearcoat = 0.5f;
    diffuse_bright.ior = 1.45f;

/* LIGHTS */
    light_data simple_light{
            light_data::type::MESH,
            vec3{1.0f, 1.0f, 1.0f},
            1.0f
    };

    camera scene_camera_1{
            vec3{3.3f, 1.0f, 0.0f},
            vec3{0.0f, 1.0f, 0.0f},
            vec3{0.0f, 1.0f, 0.0f},
            45.0f
    };
    camera scene_camera_2{
            vec3{2.0f, 1.2f, 0.0f},
            vec3{0.0f, 0.5f, 0.0f},
            vec3{0.0f, 1.0f, 0.0f},
            50.0f
    };
    camera scene_camera_3{
            vec3{5.0f, 3.0f, 0.0f},
            vec3{  0.0f, 0.75f, 0.0f},
            vec3{  0.0f, 1.0f, 0.0f},
            30.0f
    };
    camera scene_camera_4{
            vec3{5.0f, 3.0f, 0.0f},
            vec3{  0.0f, 0.75f, 0.0f},
            vec3{  0.0f, 1.0f, 0.0f},
            30.0f
    };
    camera scene_camera_5{
            vec3{5.0f, 3.0f, 0.0f},
            vec3{  0.0f, 0.75f, 0.0f},
            vec3{  0.0f, 1.0f, 0.0f},
            30.0f
    };

    std::vector<std::tuple<std::string, camera*>> scene_meta{
            {"cornell-box-w-boxes", &scene_camera_1},
            {"dragon", &scene_camera_2},
            {"mitsuba", &scene_camera_3},
            {"suzanne", &scene_camera_4},
            {"three-sphere-test", &scene_camera_5}
    };


#pragma endregion

    std::vector<std::tuple<std::string, material_data>> mats{
            {"diffuse_dark",       diffuse_dark},
            {"diffuse_bright",     diffuse_bright},
            {"disney_diffuse",     disney_diffuse},
            {"disney_subsurface",  disney_subsurface},
            {"diseny_retro",       diseny_retro},
            {"diseny_sheen",       diseny_sheen},
            {"diseny_clearcoat_glossy", diseny_clearcoat_glossy},
            {"diseny_clearcoat_rough", diseny_clearcoat_rough},
            {"diseny_microfacet_glossy", diseny_microfacet_glossy},
            {"diseny_microfacet_rough", diseny_microfacet_rough},
            {"diffuse_brdf",       diffuse_brdf}
    };
    std::vector<std::tuple<std::string, light_data>> li{
            {"simple_light", simple_light},
    };
    std::vector<entity> entities{};

    std::string scene_name;
    camera cam{};

    int k{0};
    for (auto const& [name, camera]: scene_meta)
    {
        fmt::print(fg(color::start), "SCENE[{}]: {}\n", k++, name);
    }

    std::getline(std::cin, scene_name);

    if (scene_name.empty())
    {
        scene_name = std::get<std::string>(scene_meta[1]);
        cam = *std::get<camera*>(scene_meta[1]);
    } else
    {
        int s = std::stoi(scene_name);
        scene_name = std::get<std::string>(scene_meta[s]);
        cam = *std::get<camera*>(scene_meta[s]);
    }

    auto const meshes{
            load_obj(fmt::format("{}/{}{}{}", prefix_path, "scenes/", scene_name, ".obj"))};

    std::string filename;
    fmt::print(fg(color::log), "ENTER IMAGE NAME: ");
    std::getline(std::cin, filename);

    fmt::print(fg(color::log), "> MATERIALS\n");
    for (uint32_t i{0}; i < mats.size(); i++)
        fmt::print("{} [{}]\n", std::get<std::string>(mats[i]), i);

    for (auto& [name, mesh]: meshes)
    {
        fmt::print("{}", name);
        std::string in;
        std::getline(std::cin, in);
        if (!in.empty())
            entities.push_back({std::stoi(in)});
    }


    fmt::print(fg(color::log), "> LIGHTS\n");
    for (uint32_t i{0}; i < li.size(); i++)
        fmt::print("{} [{}]\n", std::get<std::string>(li[i]), i);


    for (auto& [name, mesh]: meshes)
    {
        fmt::print("{}", name);
        std::string in;
        std::getline(std::cin, in);
        if (!in.empty())
            entities[entities.size() - 1].lightId = std::stoi(in);
    }

    optix_init();
    optix_ray_gen_program();
    optix_miss_program();
    optix_launch_params();
    optix_triangle_geom();

    /* SCENE SELECT */

    od.buffer_size = ivec2{1024};
    od.frame_buffer = create_pinned_host_buffer(
            od.context, OWL_INT, od.buffer_size.x * od.buffer_size.y);
    od.use_environment_map = true;
    od.max_samples = 1024;
    od.max_path_depth = 64;

    /* RENDER */
    uint64_t i{0};
    for (auto& [n, m]: meshes)
        add(m.get(), entities[i++]);

    std::vector<material_data> materials{};
    for (auto& e: mats)
        materials.push_back(std::get<material_data>(e));

    std::vector<light_data> lights{};
    for (auto& e: li)
        lights.push_back(std::get<light_data>(e));


    render(to_camera_data(cam, od.buffer_size), materials, lights);

    // copy image buffer

    image_buffer result{od.buffer_size.x, od.buffer_size.y,
                        (uint32_t*) buffer_to_pointer(od.frame_buffer, 0), image_buffer::tag::referenced};
    write_image(result, fmt::format("{}/{}.png", prefix_path, filename));

    optix_destroy();
}
