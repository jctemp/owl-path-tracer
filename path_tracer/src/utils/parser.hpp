
#ifndef PATH_TRACER_PARSER_HPP
#define PATH_TRACER_PARSER_HPP

#include "camera.hpp"
#include "device/device_global.hpp"

#include <vector>
#include <string>
#include <tuple>

std::vector<std::tuple<std::string, material_data>> parse_materials(std::string const& config_path);

camera_data parse_camera(std::string const& config_path, ivec2 framebuffer_size);

#endif // !PATH_TRACER_PARSER_HPP
