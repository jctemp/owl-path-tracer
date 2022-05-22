
#ifndef PATH_TRACER_DEVICE_GLOBAL_HPP
#define PATH_TRACER_DEVICE_GLOBAL_HPP

#include <cuda_runtime.h>
#include "types.hpp"
#include "camera.hpp"

enum class material_type
{
    none = 0 << 0,
    lambertian = 1 << 0,
    diffuse = 1 << 2,
    specular = 1 << 3,
    clearcoat = 1 << 4
};

struct material_data
{
    vec3 base_color{0.8f, 0.8f, 0.8f};
    float subsurface{0.0f};
    vec3 subsurface_radius{1.0f, 0.2f, 0.1f};
    vec3 subsurface_color{0.8f, 0.8f, 0.8f};
    float metallic{0.0f};
    float specular{0.5f};
    float specular_tint{1.0f};
    float roughness{0.5f};
    float sheen{0.0f};
    float sheen_tint{1.0f};
    float clearcoat{0.0f};
    float clearcoat_gloss{0.03f};
    float ior{1.45f};
    float transmission{0.0f};
    float transmission_roughness{0.0f};
};

struct light_data
{
    enum class type
    {
        none = 0 << 0,
        mesh = 1 << 0
    };

    type type{type::mesh};
    vec3 color{1.f};
    float intensity{1.f};
    float exposure{0.f};
    float falloff{2.0f};
    bool use_surface_area{false};
};

struct launch_params_data
{
    int32_t max_path_depth;
    int32_t max_samples;
    Buffer material_buffer;
    Buffer light_buffer;
    OptixTraversableHandle world;
    cudaTextureObject_t environment_map;
    bool use_environment_map;
};

struct triangle_geom_data
{
    int32_t matId{-1};
    int32_t lightId{-1};
    ivec3* index{};
    vec3* vertex{};
    vec3* normal{};
};

struct ray_gen_data
{
    uint32_t* fb_ptr;
    ivec2 fb_size;
    camera_data camera;
};

#endif //PATH_TRACER_DEVICE_GLOBAL_HPP
