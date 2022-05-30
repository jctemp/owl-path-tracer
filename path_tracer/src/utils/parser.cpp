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

        data.base_color = {
                material["base_color"][0].get<float>(),
                material["base_color"][1].get<float>(),
                material["base_color"][2].get<float>()};
        data.subsurface = material["subsurface"].get<float>();
        data.metallic = material["metallic"].get<float>();
        data.specular = material["specular"].get<float>();
        data.specular_tint = material["specular_tint"].get<float>();
        data.roughness = material["roughness"].get<float>();
        data.anisotropic = material["anisotropic"].get<float>();
        data.sheen = material["sheen"].get<float>();
        data.sheen_tint = material["sheen_tint"].get<float>();
        data.clearcoat = material["clearcoat"].get<float>();
        data.clearcoat_gloss = material["clearcoat_gloss"].get<float>();
        data.ior = material["ior"].get<float>();
        data.specular_transmission = material["specular_transmission"].get<float>();
        data.specular_transmission_roughness = material["specular_transmission_roughness"].get<float>();

        materials.emplace_back(material["name"], data);
    }

    return materials;
}

camera_data parse_camera(std::string const& config_path, ivec2 framebuffer_size)
{
    fmt::print(fg(color::log), "Parsing camera\n");

    auto config{load_json(config_path)};
    auto const cam{config["camera"]};
    camera data{};

    data.look_from = {cam["look_from"][0].get<float>(), cam["look_from"][1].get<float>(), cam["look_from"][2].get<float>()};
    data.look_at = {cam["look_at"][0].get<float>(), cam["look_at"][1].get<float>(), cam["look_at"][2].get<float>()};
    data.look_up = {cam["look_up"][0].get<float>(), cam["look_up"][1].get<float>(), cam["look_up"][2].get<float>()};
    data.vertical_fov = cam["vertical_fov"].get<float>();

    return to_camera_data(data, framebuffer_size);
}
