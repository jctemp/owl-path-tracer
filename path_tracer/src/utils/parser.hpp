
#ifndef PATH_TRACER_PARSER_HPP
#define PATH_TRACER_PARSER_HPP

#include "camera.hpp"
#include "device/device_global.hpp"

#include <vector>
#include <string>
#include <tuple>

struct test_data
{
    std::string name{};
    std::string material_name{};
    std::string attribute_name{};

    material_type material_type{};

    std::vector<vec3> vec_values{};
    std::vector<float> flt_values{};
    float step_size{};
};

struct settings_data
{
    std::string scene{};
    ivec2 buffer_size{};
    int32_t max_samples{};
    int32_t max_path_depth{};
    bool environment_use{};
    bool environment_auto{};
    vec3 environment_color{};
    float environment_intensity{};

    test_data test{};
};

std::vector<std::tuple<std::string, material_data, std::string>> parse_materials(std::string const& config_path);

camera_data parse_camera(std::string const& config_path, ivec2 framebuffer_size);

settings_data parse_settings(std::string const& config_path);

#endif // !PATH_TRACER_PARSER_HPP
