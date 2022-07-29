
#ifndef PATH_TRACER_MESH_LOADER_HPP
#define PATH_TRACER_MESH_LOADER_HPP

#include "types.hpp"
#include <tuple>
#include <vector>
#include <string>
#include <memory>

struct mesh
{
    std::vector<ivec3> indices;
    std::vector<vec3> vertices;
    std::vector<vec3> normals;
    std::vector<vec2> texcoords;
};

extern std::vector<std::tuple<std::string, std::shared_ptr<mesh>>> load_obj(std::string const& obj_file);

#endif //PATH_TRACER_MESH_LOADER_HPP
