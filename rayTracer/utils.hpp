#ifndef UTILS_HPP
#define UTILS_HPP

#include "deviceCode.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <string>
#include <ostream>
#include <map>
#include <owl/common/math/vec.h>
#include <simpleLogger.hpp>

using vec3f = owl::vec3f;
using vec3i = owl::vec3i;

int32_t constexpr TRI_SIZE{ 3 };
struct TrianglesMesh
{
	std::vector<vec3i> index;
	std::vector<vec3f> vertex;
};

std::string objPath{ "C:\\Users\\jamie\\Desktop\\Dragon.obj" };
std::string mtlPath{ "C:\\Users\\jamie\\Desktop\\" };

void createMesh(TrianglesMesh* mesh, tinyobj::shape_t const &shape, tinyobj::attrib_t const& attrib)
{
	// global, no offset, shared by all meshes
	auto& vertices{ attrib.vertices };
	auto& normals{ attrib.normals };

	// per mesh data with global indices
	auto& indices{ shape.mesh.indices };

	// mapping of global ids to local ids
	std::map<int32_t, int32_t> globalToLocal{};

	std::size_t indexOffset{ 0 };
	for (std::size_t f{ 0 }; f < shape.mesh.num_face_vertices.size(); ++f)
	{
		vec3i localFace{};
		for (size_t v{ 0 }; v < TRI_SIZE; v++)
		{
			// get vertex, normal and uv ID
			tinyobj::index_t idx{ shape.mesh.indices[indexOffset + v] };

			// set global vertex ID
			int32_t vertexGlobalID{ idx.vertex_index };

			// check if global ID is mapped
			if (!globalToLocal.contains(vertexGlobalID))
			{
				globalToLocal.insert({ vertexGlobalID, int32_t(mesh->vertex.size())});
				mesh->vertex.emplace_back(
					attrib.vertices[3 * size_t(vertexGlobalID) + 0],
					attrib.vertices[3 * size_t(vertexGlobalID) + 1],
					attrib.vertices[3 * size_t(vertexGlobalID) + 2]
				);
			}
			localFace[v] = globalToLocal[vertexGlobalID];
		}
		mesh->index.push_back(localFace);
		indexOffset += TRI_SIZE;
	}

	//// loop over triangles
	//for (std::size_t i{ 0 }; i < indices.size(); i += 3)
	//{
	//	mesh->index.emplace_back(
	//		indices[i + 0].vertex_index,
	//		indices[i + 1].vertex_index,
	//		indices[i + 2].vertex_index
	//	);
	//}
	//for (std::size_t i{ 0 }; i < vertices.size(); i += 3)
	//{
	//	mesh->vertex.emplace_back(
	//		vertices[i + 0],
	//		vertices[i + 1],
	//		vertices[i + 2]
	//	);
	//}

	SL_LOG("Create Mesh");
}

std::vector<TrianglesMesh*> loadOBJ()
{
	// creater OBJ reader
	tinyobj::ObjReader reader{};

	tinyobj::ObjReaderConfig readerConfig{};
	readerConfig.mtl_search_path = mtlPath;
	readerConfig.triangulate = false;

	// check for errors
	if (!reader.ParseFromFile(objPath, readerConfig))
	{
		if (!reader.Error().empty())
		{
			SL_ERROR(reader.Error());
		}
		exit(1);
	}

	if (!reader.Warning().empty())
	{
		SL_WARN(reader.Warning());
	}

	// get references to attrib and shapes of reader
	tinyobj::attrib_t const& attrib{ reader.GetAttrib() };
	std::vector<tinyobj::shape_t> const& shapes{ reader.GetShapes() };

	std::vector<TrianglesMesh*> meshes{};
	for (auto& shape : shapes)
	{
		TrianglesMesh* mesh{ new TrianglesMesh{} };
		createMesh(mesh, shape, attrib);
		meshes.push_back(mesh);
	}

	return meshes;
}
#endif // !UTILS_HPP
