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
        data.emission = material["emission"].get<float>();

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

settings_data parse_settings(std::string const& config_path)
{
fmt::print(fg(color::log), "Parsing settings\n");

    auto config{load_json(config_path)};
    settings_data data{};

    auto test{config["test"]};
    data.test.name = test["name"].get<std::string>();
    data.test.material_name = test["material_name"].get<std::string>();
    data.test.attribute_name = test["attribute_name"].get<std::string>();

    data.test.material_type = static_cast<material_type>(test["material_type"].get<int32_t>());

    data.test.step_size = test["step_size"].get<float>();
    for (auto& value: test["values"])
    {
        if (value.is_array())
            data.test.vec_values.emplace_back(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
        else
            data.test.flt_values.emplace_back(value.get<float>());
    }

    data.scene = config["scene"].get<std::string>();
    data.buffer_size = {config["buffer_size"][0].get<int>(), config["buffer_size"][1].get<int>()};
    data.max_path_depth = config["max_path_depth"].get<int32_t>();
    data.max_samples = config["max_samples"].get<int32_t>();
    data.environment_use = config["environment_use"].get<bool>();
    data.environment_auto = config["environment_auto"].get<bool>();
    data.environment_color = {
            config["environment_color"][0].get<float>(),
            config["environment_color"][1].get<float>(),
            config["environment_color"][2].get<float>()};
    data.environment_intensity = config["environment_intensity"].get<float>();

    return data;
}

