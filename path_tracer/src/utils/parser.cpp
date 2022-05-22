#include "parser.hpp"

#include "types.hpp"

#include <nlohmann/json.hpp>
#include <fstream>

using namespace nlohmann;

json load_json(std::string const& file_path)
{
    nlohmann::json config{};
    std::ifstream config_file{file_path};
    config_file >> config;
    config_file.close();
    return config;
}

std::vector<std::tuple<std::string, material_data>> parse_materials(std::string const& config_path)
{
    fmt::print(fg(color::log), "Parsing materials\n");

    auto config{load_json(config_path)};
    auto materials{std::vector<std::tuple<std::string, material_data>>{}};

    for (auto& material: config["materials"])
    {
        fmt::print(" - {}\n", material["name"]);
        material_data data{};

        data.base_color = {material["base_color"][0], material["base_color"][1], material["base_color"][2]};
        data.subsurface = material["subsurface"];
        data.subsurface_radius = {material["subsurface_radius"][0], material["subsurface_radius"][1],
                                  material["subsurface_radius"][2]};
        data.subsurface_color = {material["subsurface_color"][0], material["subsurface_color"][1],
                                 material["subsurface_color"][2]};
        data.metallic = material["metallic"];
        data.specular = material["specular"];
        data.specular_tint = material["specular_tint"];
        data.roughness = material["roughness"];
        data.sheen = material["sheen"];
        data.sheen_tint = material["sheen_tint"];
        data.clearcoat = material["clearcoat"];
        data.clearcoat_gloss = material["clearcoat_gloss"];
        data.ior = material["ior"];

        materials.emplace_back(material["name"], data);
    }

    return materials;
}

std::vector<std::tuple<std::string, camera>> parse_scenes(std::string const& config_path)
{
    fmt::print(fg(color::log), "Parsing scenes\n");

    auto config{load_json(config_path)};
    auto scenes{std::vector<std::tuple<std::string, camera>>{}};

    for (auto& scene: config["scenes"])
    {
        fmt::print(" - {}\n", scene["scene"]);
        camera data{};

        data.look_from = {scene["look_from"][0], scene["look_from"][1], scene["look_from"][2]};
        data.look_at = {scene["look_at"][0], scene["look_at"][1], scene["look_at"][2]};
        data.look_up = {scene["look_up"][0], scene["look_up"][1], scene["look_up"][2]};
        data.vertical_fov = scene["vertical_fov"];

        scenes.emplace_back(scene["scene"], data);
    }

    return scenes;
}
