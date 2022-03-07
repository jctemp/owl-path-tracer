#ifndef OBJ_LOADER_HPP
#define OBJ_LOADER_HPP

#include "hostCode.hpp"

#include <string>
#include <vector>

#include <owl/common/math/vec.h>

namespace ba
{
	/// <summary>
	/// TrianglesMesh is a data transfer user-defined data type.
	/// Its purpose is to transfer data of meshes to the calling system.
	/// </summary>
	struct TrianglesMesh : public Mesh
	{
		std::vector<owl::vec3i> index;
		std::vector<owl::vec3f> vertex;
	};

	/// <summary>
	/// This function loads a given obj-file and creates a vector
	/// of pointers to TriangleMesh. See TrianglesMesh for more detail
	/// about the saved data.
	/// The POINTERS are heap allocated. The caller must FREE the mem.
	/// </summary>
	/// <param name="pathToObj">a relative or absolute path to an obj-file</param>
	/// <returns>a vector with points of TriangleMesh</returns>
	extern std::vector<TrianglesMesh*> loadOBJ(std::string const& pathToObj);
}

#endif // !OBJ_LOADER_HPP
