
#ifndef PATH_TRACER_DEVICE_GLOBAL_HPP
#define PATH_TRACER_DEVICE_GLOBAL_HPP

#include <cuda_runtime.h>
#include "types.hpp"
#include "camera.hpp"

struct material_data
{
    enum class type
    {
        none = 0 << 0,
        lambert = 1 << 1,
        disney = 1 << 2,
        disney_diffuse = 1 << 3,
        disney_subsurface = 1 << 4,
        disney_retro = 1 << 5,
        disney_sheen = 1 << 6,
        disney_clearcoat = 1 << 7,
        disney_microfacet = 1 << 8
    };

    type type{ type::none };
    vec3  base_color{0.8f, 0.8f, 0.8f };
    float subsurface{ 0.0f };
    vec3  subsurface_radius{1.0f, 0.2f, 0.1f };
    vec3  subsurface_color{0.8f, 0.8f, 0.8f };
    float metallic{ 0.0f };
    float specular{ 0.5f };
    float specular_tint{1.0f };
    float roughness{ 0.5f };
    float sheen{ 0.0f };
    float sheen_tint{1.0f };
    float clearcoat{ 0.0f };
    float clearcoat_gloss{0.03f };
    float ior{ 1.45f };
};

struct light_data
{
    enum class type
    {
        NONE = 0 << 0,
        MESH = 1 << 0
    };

    type type{ type::MESH };
    vec3 color{ 1.f };
    float intensity{ 1.f };
    float exposure{ 0.f };
    float falloff{ 2.0f };
    bool use_surface_area = false;
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
    int32_t matId{ -1 };
    int32_t lightId{ -1 };
    ivec3* index;
    vec3* vertex;
    vec3* normal;
};

struct ray_gen_data
{
    uint32_t* fbPtr;
    ivec2 fbSize;
    camera_data camera;
};

struct miss_data
{
};

#endif //PATH_TRACER_DEVICE_GLOBAL_HPP
