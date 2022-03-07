#ifndef BA_RENDERER_HPP
#define BA_RENDERER_HPP

#include <vector>

#include <glm/vec3.hpp>

namespace ba
{
    /// <summary>
    /// TrianglesMesh is a data transfer user-defined data type.
    /// Its purpose is to transfer data of meshes to the calling system.
    /// </summary>
    struct Mesh
    {
        std::vector<glm::i32vec3> index;
        std::vector<glm::vec3> vertex;
    };

    struct Renderer
    {
        /// <summary>
        /// Initialises the renderer with all necessary values to be ready.
        /// Renderer is ready to receive data and render it after success.
        /// </summary>
        /// <returns>0 in case of success otherwise different</returns>
        virtual int init() = 0;

        /// <summary>
        /// Releases all resources of the renderer which are currently in
        /// use.
        /// </summary>
        /// <returns>0 in case of success otherwise different</returns>
        virtual int release() = 0;

        /// <summary>
        /// Set the meta data of the renderer. 
        /// 
        /// TODO: struct with information => intermediate state
        /// </summary>
        /// <returns></returns>
        virtual int renderSetting() = 0;

        /// <summary>
        /// Takes an intermediate form of a mesh and makes it ready for the
        /// renderer. After loading successful the mesh can be rendered.
        /// </summary>
        /// <param name="m">An object of the type Mesh</param>
        /// <returns>0 in case of success otherwise different</returns>
        virtual int add(Mesh* m) = 0;

        /// <summary>
        /// Renderes the Meshes with the specifed render settings
        /// </summary>
        /// <returns>0 in case of success otherwise different</returns>
        virtual int render() = 0;
    
        virtual uint32_t const* fbPtr() const = 0;
    
    };

}

#endif // !BA_RENDERER_HPP
