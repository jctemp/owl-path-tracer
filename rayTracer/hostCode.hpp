#ifndef BA_HOST_CODE_HPP
#define BA_HOST_CODE_HPP

namespace ba
{

    struct Mesh
    {
        enum class Type
        {
            TRIANGLE
        };
        Type t;
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

        virtual int renderSetting() = 0;

        /// <summary>
        /// Takes an intermediate form of a mesh and makes it ready for the
        /// renderer. After loading successful the mesh can be rendered.
        /// </summary>
        /// <param name="m">An object of the type Mesh</param>
        /// <returns>0 in case of success otherwise different</returns>
        virtual int load(Mesh* m) = 0;

        /// <summary>
        /// Renderes the Meshes with the specifed render settings
        /// </summary>
        /// <returns>0 in case of success otherwise different</returns>
        virtual int render() = 0;
    };
}

#endif // !BA_HOST_CODE_HPP
