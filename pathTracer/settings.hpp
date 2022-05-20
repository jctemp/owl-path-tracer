
#ifndef PATH_TRACER_SETTINGS_HPP
#define PATH_TRACER_SETTINGS_HPP

#include "device/device_global.hpp"
#include <string>
#include <vector>
#include <tuple>

/* BSDF */
static material_data diffuse_dark{material_data::type::disney};
static material_data diffuse{material_data::type::disney};
static material_data diffuse_with_sheen{material_data::type::disney};
static material_data subsurface{material_data::type::disney};
static material_data glossy{material_data::type::disney};
static material_data metal{material_data::type::disney};
static material_data rough_metal{material_data::type::disney};
static material_data plastic{material_data::type::disney};
static material_data rough_plastic{material_data::type::disney};

/* LIGHTS */
static light_data simple_light{};

static camera scene_camera_1{};
static camera scene_camera_2{};
static camera scene_camera_3{};
static camera scene_camera_4{};
static camera scene_camera_5{};

static std::vector<std::tuple<std::string, camera*>> scene_meta{
    {"cornell-box-w-boxes", &scene_camera_1},
    {"dragon", &scene_camera_2},
    {"mitsuba", &scene_camera_3},
    {"suzanne", &scene_camera_4},
    {"three-sphere-test", &scene_camera_5}
};

void prepare()
{
    diffuse_dark.base_color = vec3{0.2f};
    diffuse_dark.subsurface = 0.0f;
    diffuse_dark.subsurface_radius = vec3{1.0f, 0.2f, 0.1f};
    diffuse_dark.subsurface_color = vec3{0.8f, 0.8f, 0.8f};
    diffuse_dark.metallic = 0.0f;
    diffuse_dark.specular = 0.0f;
    diffuse_dark.specular_tint = 1.0f;
    diffuse_dark.roughness = 0.5f;
    diffuse_dark.sheen = 0.0f;
    diffuse_dark.sheen_tint = 1.0f;
    diffuse_dark.clearcoat = 0.0f;
    diffuse_dark.clearcoat_gloss = 0.03f;
    diffuse_dark.ior = 1.45f;

    diffuse.base_color = vec3{0.8f};
    diffuse.subsurface = 0.0f;
    diffuse.subsurface_radius = vec3{1.0f, 0.2f, 0.1f};
    diffuse.subsurface_color = vec3{0.8f, 0.8f, 0.8f};
    diffuse.metallic = 0.0f;
    diffuse.specular = 0.0f;
    diffuse.specular_tint = 1.0f;
    diffuse.roughness = 0.5f;
    diffuse.sheen = 0.0f;
    diffuse.sheen_tint = 1.0f;
    diffuse.clearcoat = 0.0f;
    diffuse.clearcoat_gloss = 0.03f;
    diffuse.ior = 1.45f;

    diffuse_with_sheen.base_color = vec3{0.8f};
    diffuse_with_sheen.subsurface = 0.0f;
    diffuse_with_sheen.subsurface_radius = vec3{1.0f, 0.2f, 0.1f};
    diffuse_with_sheen.subsurface_color = vec3{0.8f, 0.8f, 0.8f};
    diffuse_with_sheen.metallic = 0.0f;
    diffuse_with_sheen.specular = 0.0f;
    diffuse_with_sheen.specular_tint = 1.0f;
    diffuse_with_sheen.roughness = 0.5f;
    diffuse_with_sheen.sheen = 0.8f;
    diffuse_with_sheen.sheen_tint = 1.0f;
    diffuse_with_sheen.clearcoat = 0.0f;
    diffuse_with_sheen.clearcoat_gloss = 0.03f;
    diffuse_with_sheen.ior = 1.45f;

    subsurface.base_color = vec3{0.8f};
    subsurface.subsurface = 0.8f;
    subsurface.subsurface_radius = vec3{1.0f, 0.2f, 0.1f};
    subsurface.subsurface_color = vec3{0.8f, 0.8f, 0.8f};
    subsurface.metallic = 0.0f;
    subsurface.specular = 0.0f;
    subsurface.specular_tint = 1.0f;
    subsurface.roughness = 0.5f;
    subsurface.sheen = 0.0f;
    subsurface.sheen_tint = 1.0f;
    subsurface.clearcoat = 0.0f;
    subsurface.clearcoat_gloss = 0.03f;
    subsurface.ior = 1.45f;

    glossy.base_color = vec3{0.8f};
    glossy.subsurface = 0.0f;
    glossy.subsurface_radius = vec3{1.0f, 0.2f, 0.1f};
    glossy.subsurface_color = vec3{0.8f, 0.8f, 0.8f};
    glossy.metallic = 0.0f;
    glossy.specular = 0.8f;
    glossy.specular_tint = 1.0f;
    glossy.roughness = 0.5f;
    glossy.sheen = 0.0f;
    glossy.sheen_tint = 1.0f;
    glossy.clearcoat = 0.0f;
    glossy.clearcoat_gloss = 0.03f;
    glossy.ior = 1.45f;

    metal.base_color = vec3{0.8f};
    metal.subsurface = 0.0f;
    metal.subsurface_radius = vec3{1.0f, 0.2f, 0.1f};
    metal.subsurface_color = vec3{0.8f, 0.8f, 0.8f};
    metal.metallic = 0.8f;
    metal.specular = 0.0f;
    metal.specular_tint = 1.0f;
    metal.roughness = 0.0f;
    metal.sheen = 0.0f;
    metal.sheen_tint = 1.0f;
    metal.clearcoat = 0.0f;
    metal.clearcoat_gloss = 0.03f;
    metal.ior = 1.45f;

    rough_metal.base_color = vec3{0.8f};
    rough_metal.subsurface = 0.0f;
    rough_metal.subsurface_radius = vec3{1.0f, 0.2f, 0.1f};
    rough_metal.subsurface_color = vec3{0.8f, 0.8f, 0.8f};
    rough_metal.metallic = 0.8f;
    rough_metal.specular = 0.0f;
    rough_metal.specular_tint = 1.0f;
    rough_metal.roughness = 0.8f;
    rough_metal.sheen = 0.0f;
    rough_metal.sheen_tint = 1.0f;
    rough_metal.clearcoat = 0.0f;
    rough_metal.clearcoat_gloss = 0.03f;
    rough_metal.ior = 1.45f;

    plastic.base_color = vec3{0.8f};
    plastic.subsurface = 0.0f;
    plastic.subsurface_radius = vec3{1.0f, 0.2f, 0.1f};
    plastic.subsurface_color = vec3{0.8f, 0.8f, 0.8f};
    plastic.metallic = 0.0f;
    plastic.specular = 0.0f;
    plastic.specular_tint = 1.0f;
    plastic.roughness = 0.5f;
    plastic.sheen = 0.0f;
    plastic.sheen_tint = 1.0f;
    plastic.clearcoat = 0.8f;
    plastic.clearcoat_gloss = 0.03f;
    plastic.ior = 1.45f;

    rough_plastic.base_color = vec3{0.8f};
    rough_plastic.subsurface = 0.0f;
    rough_plastic.subsurface_radius = vec3{1.0f, 0.2f, 0.1f};
    rough_plastic.subsurface_color = vec3{0.8f, 0.8f, 0.8f};
    rough_plastic.metallic = 0.0f;
    rough_plastic.specular = 0.0f;
    rough_plastic.specular_tint = 1.0f;
    rough_plastic.roughness = 0.8f;
    rough_plastic.sheen = 0.0f;
    rough_plastic.sheen_tint = 1.0f;
    rough_plastic.clearcoat = 0.8f;
    rough_plastic.clearcoat_gloss = 0.03f;
    rough_plastic.ior = 1.45f;

    simple_light.intensity = 10.0f;

    scene_camera_1.look_from = vec3{3.3f, 1.0f, 0.0f};
    scene_camera_1.look_at = vec3{0.0f, 1.0f, 0.0f};
    scene_camera_1.look_up = vec3{0.0f, 1.0f, 0.0f};
    scene_camera_1.vertical_fov = 45.0f;

    scene_camera_2.look_from = vec3{2.0f, 1.2f, 0.0f};
    scene_camera_2.look_at = vec3{0.0f, 0.5f, 0.0f};
    scene_camera_2.look_up = vec3{0.0f, 1.0f, 0.0f};
    scene_camera_2.vertical_fov = 50.0f;

    scene_camera_3.look_from = vec3{5.0f, 3.0f, 0.0f};
    scene_camera_3.look_at = vec3{  0.0f, 0.75f, 0.0f};
    scene_camera_3.look_up = vec3{  0.0f, 1.0f, 0.0f};
    scene_camera_3.vertical_fov = 30.0f;

    scene_camera_4.look_from = vec3{5.0f, 3.0f, 0.0f};
    scene_camera_4.look_at = vec3{  0.0f, 0.75f, 0.0f};
    scene_camera_4.look_up = vec3{  0.0f, 1.0f, 0.0f};
    scene_camera_4.vertical_fov = 30.0f;

    scene_camera_5.look_from = vec3{5.0f, 3.0f, 0.0f};
    scene_camera_5.look_at = vec3{  0.0f, 0.75f, 0.0f};
    scene_camera_5.look_up = vec3{  0.0f, 1.0f, 0.0f};
    scene_camera_5.vertical_fov = 30.0f;

}

#endif //PATH_TRACER_SETTINGS_HPP
