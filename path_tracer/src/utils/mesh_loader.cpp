#include "mesh_loader.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#define    OBJ_TRIANGLE 3

#include <filesystem>
#include <tiny_obj_loader.h>

mesh* create_mesh(tinyobj::shape_t const& shape, tinyobj::attrib_t const& attribute)
{
    auto const mesh_ptr{new mesh{}};

    // global, no offset, shared by all meshes
    auto const& mesh_vertices{attribute.vertices};
    auto const& mesh_normals{attribute.normals};
    auto const& mesh_texcoords{attribute.texcoords};
  
    // per mesh data with global indices
    auto& indices{shape.mesh.indices};

    // mapping of global ids to local ids
    std::map<int32_t, int32_t> vertex_mapping{};

    // num_face_vertices gives the amount of faces + how many vertices per face
    std::size_t index_offset{0};

    for (std::size_t f{0}; f < shape.mesh.num_face_vertices.size(); ++f)
    {
        mesh_ptr->indices.emplace_back(0, 0, 0);
        auto& vertex_local{mesh_ptr->indices[mesh_ptr->indices.size() - 1]};

        for (size_t v{0}; v < OBJ_TRIANGLE; v++)
        {
            // get vertex, normal and uv ID
            tinyobj::index_t const index{indices[index_offset + v]};

            // set global vertex ID
            int32_t const vertex_id{index.vertex_index};
            int32_t const normal_id{index.normal_index};
            int32_t const texcoord_id{ index.texcoord_index };

            // check if global ID is mapped
            if (!vertex_mapping.contains(vertex_id))
            {
                vertex_mapping.insert({vertex_id, static_cast<int32_t>(mesh_ptr->vertices.size())});
                mesh_ptr->vertices.emplace_back(
                        mesh_vertices[3 * static_cast<uint64_t>(vertex_id) + 0],
                        mesh_vertices[3 * static_cast<uint64_t>(vertex_id) + 1],
                        mesh_vertices[3 * static_cast<uint64_t>(vertex_id) + 2]
                );
            }
            vertex_local[v] = vertex_mapping[vertex_id];

            // add normals
            if (index.normal_index >= 0)
            {
                // wtf no idea why this works lol
                while (mesh_ptr->normals.size() < mesh_ptr->vertices.size())
                {
                    mesh_ptr->normals.emplace_back(
                            mesh_normals[3 * static_cast<uint64_t>(normal_id) + 0],
                            mesh_normals[3 * static_cast<uint64_t>(normal_id) + 1],
                            mesh_normals[3 * static_cast<uint64_t>(normal_id) + 2]
                    );
                }
            }

            if (index.texcoord_index >= 0)
            {
                // wtf no idea why this works lol
                while (mesh_ptr->texcoords.size() < mesh_ptr->vertices.size())
                {
                    mesh_ptr->texcoords.emplace_back(
                        mesh_texcoords[2 * static_cast<uint64_t>(texcoord_id) + 0],
                        mesh_texcoords[2 * static_cast<uint64_t>(texcoord_id) + 1]
                    );
                }
            }
        }
        index_offset += OBJ_TRIANGLE;
    }
    return mesh_ptr;
}

std::vector<std::tuple<std::string, std::shared_ptr<mesh>>> load_obj(std::string const& obj_file)
{
    // 1) create OBJ reader
    tinyobj::ObjReader reader{};

    tinyobj::ObjReaderConfig readerConfig{};
    readerConfig.triangulate = false;

    auto path = std::filesystem::absolute(obj_file);

    // 2) load obj file and checking for errors
    if (!reader.ParseFromFile(obj_file, readerConfig))
        if (!reader.Error().empty())
            throw std::runtime_error(reader.Error());

    // 3) check for warnings
    if (!reader.Warning().empty())
        printf("WARNING: %s\n", reader.Warning().c_str());

    // 4) get references to attribute and shapes of reader
    /*
    > tinyobj::attrib_t contains for all object vertex, normals and uv data
    > hence the indices inside shapes refers to the global index
    >
    > tinyobj::shape_t has metadata of object. It helps to build faces
    > lines and points. Creating custom meshes needs an internal mapping
    */
    tinyobj::attrib_t const& attribute{reader.GetAttrib()};
    std::vector<tinyobj::shape_t> const& shapes{reader.GetShapes()};

    // 5.) create meshes
    std::vector<std::tuple<std::string, std::shared_ptr<mesh>>> tuples{};
    for (std::size_t i{0}; i < shapes.size(); ++i)
        tuples.emplace_back(shapes[i].name, create_mesh(shapes[i], attribute));

    return tuples;
}
