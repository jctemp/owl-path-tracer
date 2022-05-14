#include "ObjLoader.hpp"
#include <pt/Types.hpp>
#include <SimpleLogger.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#define	OBJ_TRIANGLE 3
#include <tiny_obj_loader.h>


void createMesh(Mesh* mesh, tinyobj::shape_t const& shape, tinyobj::attrib_t const& attrib)
{
	// global, no offset, shared by all meshes
	auto& vertices{ attrib.vertices };
	auto& normals{ attrib.normals };
	auto& uv{ attrib.texcoords };

	// per mesh data with global indices
	auto& indices{ shape.mesh.indices };

	// mapping of global ids to local ids
	std::map<int32_t, int32_t> vertexMapping{};

	std::size_t indexOffset{ 0 };
	// num_face_vertices gives the amount of faces + how many vertices per face
	for (std::size_t f{ 0 }; f < shape.mesh.num_face_vertices.size(); ++f)
	{
		mesh->index.emplace_back(0, 0, 0);
		auto& vertexLocal{ mesh->index[mesh->index.size() - 1] };

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
				vertexMapping.insert({ vertexID, int32_t(mesh->vertex.size()) });
				mesh->vertex.emplace_back(
					vertices[3 * size_t(vertexID) + 0],
					vertices[3 * size_t(vertexID) + 1],
					vertices[3 * size_t(vertexID) + 2]
				);
			}
			vertexLocal[v] = vertexMapping[vertexID];

			// add normals
			if (idx.normal_index >= 0)
			{
				// wtf no idea why this works lol
				while (mesh->normal.size() < mesh->vertex.size())
				{
					mesh->normal.emplace_back(
						normals[3 * size_t(normalID) + 0],
						normals[3 * size_t(normalID) + 1],
						normals[3 * size_t(normalID) + 2]
					);
				}
			}
		}
		indexOffset += OBJ_TRIANGLE;
	}
}


std::tuple<std::vector<std::string>, std::vector<Mesh*>> loadOBJ(
	std::string const& pathToObj)
{
	// 1.) create OBJ reader
	tinyobj::ObjReader reader{};

	tinyobj::ObjReaderConfig readerConfig{};
	readerConfig.triangulate = false;

	// 2.) load obj file and checking for errors
	if (!reader.ParseFromFile(pathToObj, readerConfig))
		if (!reader.Error().empty()) 
		{
			SL_ERROR(reader.Error());
			exit(1);
		}

	// 3). check for warings
	if (!reader.Warning().empty())
		SL_WARN(reader.Warning());

	// 4.) get references to attrib and shapes of reader
	/*
	// > tinyobj::attrib_t contains for all object vertex, normals and uv data
	// > hence the indices inside shapes referres to the global index
	// >
	// > tinyobj::shape_t has meta data of object. It helps to build faces
	// > lines and points. Creating custom meshes needs an internal mapping
	*/
	tinyobj::attrib_t const& attrib{ reader.GetAttrib() };
	std::vector<tinyobj::shape_t> const& shapes{ reader.GetShapes() };

	// 5.) create meshes
	std::vector<Mesh*> meshes{};
	std::vector<std::string> names{};
	for (std::size_t i{ 0 }; i < shapes.size(); ++i)
	{
		auto& shape{ shapes[i] };
		Mesh* mesh{ new Mesh{} };
		createMesh(mesh, shape, attrib);
		meshes.push_back(mesh);
		names.push_back(shape.name);
	}

	return std::make_tuple(names, meshes);
}