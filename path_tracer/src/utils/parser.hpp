
#ifndef PATH_TRACER_PARSER_HPP
#define PATH_TRACER_PARSER_HPP

#include "camera.hpp"
#include "device/device_global.hpp"

#include <vector>
#include <string>
#include <tuple>

std::vector<std::tuple<std::string, light_data>> parse_lights(std::string const& config_path);

std::vector<std::tuple<std::string, material_data>> parse_materials(std::string const& config_path);

std::vector<std::tuple<std::string, camera>> parse_scenes(std::string const& config_path);

#endif // !PATH_TRACER_PARSER_HPP
