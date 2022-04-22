#ifndef OBJ_LOADER_HPP
#define OBJ_LOADER_HPP

#include <pathTracer/Renderer.hpp>

#include <string>
#include <vector>

/// <summary>
/// This function loads a given obj-file and creates a vector
/// of pointers to TriangleMesh. See TrianglesMesh for more detail
/// about the saved data.
/// The POINTERS are heap allocated. The caller must FREE the mem.
/// </summary>
/// <param name="pathToObj">a relative or absolute path to an obj-file</param>
/// <returns>a vector with points of TriangleMesh</returns>
extern std::vector<Mesh*> loadOBJ(std::string const& pathToObj);


#endif // !OBJ_LOADER_HPP
