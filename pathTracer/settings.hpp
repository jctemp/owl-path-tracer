
#ifndef PATH_TRACER_SETTINGS_HPP
#define PATH_TRACER_SETTINGS_HPP

#include "device/device_global.hpp"

/* BSDF */
material_data diffuse{material_data::type::disney};
material_data diffuse_with_sheen{material_data::type::disney};
material_data subsurface{material_data::type::disney};
material_data glossy{material_data::type::disney};
material_data metal{material_data::type::disney};
material_data rough_metal{material_data::type::disney};
material_data plastic{material_data::type::disney};
material_data rough_plastic{material_data::type::disney};

void prepare()
{
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
}

#endif //PATH_TRACER_SETTINGS_HPP
