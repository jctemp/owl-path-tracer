#pragma once

#include <types.hpp>

#include <tuple>
#include <vector>
#include <string>
#include <memory>

struct Mesh
{
    Int materialId{ -1 };
    Int lightId{ -1 };
    std::vector<Int3> indices;
    std::vector<Float3> vertices;
    std::vector<Float3> normals;
};

//struct Mesh
//{
//    std::vector<Int3> indices;
//    std::vector<Float3> vertices;
//    std::vector<Float3> normals;
//};

extern std::vector<std::tuple<std::string, std::shared_ptr<Mesh>>> load_obj(std::string const& obj_file);