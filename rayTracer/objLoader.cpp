#define TINYOBJLOADER_IMPLEMENTATION
#define	OBJ_TRIANGLE 3

#include "Renderer.hpp"
#include "ObjLoader.hpp"

#include <map>

#include <simpleLogger.hpp>
#include <tiny_obj_loader.h>


/// <summary>
/// Local function to abstract from the creation of an TriangleMesh.
/// </summary>
/// <param name="mesh">mutable TriangleMesh</param>
/// <param name="shape">meta data shape</param>
/// <param name="attrib">global values of all meshes</param>
void createMesh(ba::Mesh* mesh, tinyobj::shape_t const& shape, tinyobj::attrib_t const& attrib)
{
	// global, no offset, shared by all meshes
	auto& vertices{ attrib.vertices };
	auto& normals{ attrib.normals };

	// per mesh data with global indices
	auto& indices{ shape.mesh.indices };

	// mapping of global ids to local ids
	std::map<int32_t, int32_t> globalToLocal{};

	std::size_t indexOffset{ 0 };
	// num_face_vertices gives the amount of faces + how many vertices per face
	for (std::size_t f{ 0 }; f < shape.mesh.num_face_vertices.size(); ++f)
	{
		mesh->index.emplace_back(0, 0, 0);
		auto& localFace{mesh->index[mesh->index.size() - 1]};
		for (size_t v{ 0 }; v < OBJ_TRIANGLE; v++)
		{
			// get vertex, normal and uv ID
			tinyobj::index_t idx{ shape.mesh.indices[indexOffset + v] };

			// set global vertex ID
			int32_t vertexGlobalID{ idx.vertex_index };

			// check if global ID is mapped
			if (!globalToLocal.contains(vertexGlobalID))
			{
				globalToLocal.insert({ vertexGlobalID, int32_t(mesh->vertex.size()) });
				mesh->vertex.emplace_back(
					attrib.vertices[3 * size_t(vertexGlobalID) + 0],
					attrib.vertices[3 * size_t(vertexGlobalID) + 1],
					attrib.vertices[3 * size_t(vertexGlobalID) + 2]
				);
			}
			localFace[v] = globalToLocal[vertexGlobalID];
		}
		indexOffset += OBJ_TRIANGLE;
	}
}

std::vector<ba::Mesh*> ba::loadOBJ(std::string const& pathToObj)
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
	for (std::size_t i{ 0 }; i < shapes.size(); ++i)
	{
		auto& shape{ shapes[i] };
		SL_LOG(fmt::format("Loading {} [{}/{}]", shape.name, i + 1, shapes.size()));
		Mesh* mesh{ new Mesh{} };
		createMesh(mesh, shape, attrib);
		meshes.push_back(mesh);
	}

	return meshes;
}