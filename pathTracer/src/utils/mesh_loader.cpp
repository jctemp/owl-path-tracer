#include "mesh_loader.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#define	OBJ_TRIANGLE 3

#include <tiny_obj_loader.h>

Mesh* create_mesh(tinyobj::shape_t const& shape, tinyobj::attrib_t const& attrib)
{
    Mesh* mesh{ new Mesh{} };

    // global, no offset, shared by all meshes
    auto& meshvertices{ attrib.vertices };
    auto& meshnormals{ attrib.normals };
    auto& meshuv{ attrib.texcoords };

    // per mesh data with global indices
    auto& indices{ shape.mesh.indices };

    // mapping of global ids to local ids
    std::map<int32_t, int32_t> vertexMapping{};

    std::size_t indexOffset{ 0 };
    // num_face_vertices gives the amount of faces + how many vertices per face
    for (std::size_t f{ 0 }; f < shape.mesh.num_face_vertices.size(); ++f)
    {
        mesh->indices.emplace_back(0, 0, 0);
        auto& vertexLocal{ mesh->indices[mesh->indices.size() - 1] };

        for (size_t v{ 0 }; v < OBJ_TRIANGLE; v++)
        {
            // get vertex, normal and uv ID
            tinyobj::index_t idx{ indices[indexOffset + v] };

            // set global vertex ID
            int32_t vertexID{ idx.vertex_index };
            int32_t normalID{ idx.normal_index };

            // check if global ID is mapped
            if (!vertexMapping.contains(vertexID))
            {
                vertexMapping.insert({ vertexID, int32_t(mesh->vertices.size()) });
                mesh->vertices.emplace_back(
                    meshvertices[3 * size_t(vertexID) + 0],
                    meshvertices[3 * size_t(vertexID) + 1],
                    meshvertices[3 * size_t(vertexID) + 2]
                );
            }
            vertexLocal[v] = vertexMapping[vertexID];

            // add normals
            if (idx.normal_index >= 0)
            {
                // wtf no idea why this works lol
                while (mesh->normals.size() < mesh->vertices.size())
                {
                    mesh->normals.emplace_back(
                        meshnormals[3 * size_t(normalID) + 0],
                        meshnormals[3 * size_t(normalID) + 1],
                        meshnormals[3 * size_t(normalID) + 2]
                    );
                }
            }
        }
        indexOffset += OBJ_TRIANGLE;
    }
    return mesh;
}

std::vector<std::tuple<std::string, std::shared_ptr<Mesh>>> load_obj(std::string const& obj_file)
{
    // 1.) create OBJ reader
    tinyobj::ObjReader reader{};

    tinyobj::ObjReaderConfig readerConfig{};
    readerConfig.triangulate = false;

    // 2.) load obj file and checking for errors
    if (!reader.ParseFromFile(obj_file, readerConfig))
        if (!reader.Error().empty())
            std::runtime_error(reader.Error());

    // 3). check for warings
    if (!reader.Warning().empty())
        printf("WARNING: %s\n", reader.Warning().c_str());

    // 4.) get references to attrib and shapes of reader
    /*
    > tinyobj::attrib_t contains for all object vertex, normals and uv data
    > hence the indices inside shapes referres to the global index
    >
    > tinyobj::shape_t has meta data of object. It helps to build faces
    > lines and points. Creating custom meshes needs an internal mapping
    */
    tinyobj::attrib_t const& attrib{ reader.GetAttrib() };
    std::vector<tinyobj::shape_t> const& shapes{ reader.GetShapes() };

    // 5.) create meshes
    std::vector<std::tuple<std::string, std::shared_ptr<Mesh>>> tuples{};
    for (std::size_t i{ 0 }; i < shapes.size(); ++i)
        tuples.emplace_back(shapes[i].name, create_mesh(shapes[i], attrib));

    return tuples;
}
