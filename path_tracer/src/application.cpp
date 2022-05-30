#include "application.hpp"

extern "C" char device_ptx[];

/// prepare all owl components
void init_data(owl_data& data)
{
    /// create context and module to prepare different components
    data.owl_context = create_context(nullptr, 1);
    data.owl_module = create_module(data.owl_context, device_ptx);


    /// create bindable data for triangles
    var_decl triangles_geom_vars
            {
                    {"mesh_index",     OWL_INT,    OWL_OFFSETOF(entity_data, mesh_index)},
                    {"material_index", OWL_INT,    OWL_OFFSETOF(entity_data, material_index)},
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
void init_world(owl_data& data, std::vector<geom>& geoms)
{
    if (geoms.empty()) throw std::runtime_error("no geometries");

    auto triangles_group{owlTrianglesGeomGroupCreate(data.owl_context, geoms.size(), geoms.data())};
    build_group_acceleration_structure(triangles_group);

    data.world = create_instance_group(data.owl_context, 1, &triangles_group);
    build_group_acceleration_structure(data.world);
}

