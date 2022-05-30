
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
    float metallic{0.0f};
    float specular{0.5f};
    float specular_tint{1.0f};
    float roughness{0.5f};
    float anisotropic{0.0f};
    float sheen{0.0f};
    float sheen_tint{1.0f};
    float clearcoat{0.0f};
    float clearcoat_gloss{0.03f};
    float ior{1.45f};
    float specular_transmission{0.0f};
    float specular_transmission_roughness{0.0f};
};

struct entity_data
{
    int32_t mesh_index{-1};
    int32_t material_index{-1};
};

struct launch_params_data
{
    int32_t max_path_depth;
    int32_t max_samples;
    Buffer material_buffer;

    Buffer vertices_buffer;
    Buffer indices_buffer;
    Buffer normals_buffer;

    OptixTraversableHandle world;
    cudaTextureObject_t environment_map;
    bool use_environment_map;
};


struct ray_gen_data
{
    uint32_t* fb_ptr;
    ivec2 fb_size;
    camera_data camera;
};

#endif //PATH_TRACER_DEVICE_GLOBAL_HPP
