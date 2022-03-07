#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <device/PathTracer.hpp>
#include <pathTracer/ObjLoader.hpp>
#include <pathTracer/Renderer.hpp>
#include <pathTracer/OWLRenderer.hpp>

#include <map>

#include <stb_image_write.h>
#include <simpleLogger.hpp>

using namespace owl;

int main(void)
{
    ba::Renderer* rPtr{ new ba::OWLRenderer{} };
    ba::Renderer &renderer = *rPtr;

    std::vector<ba::Mesh*> meshes{ ba::loadOBJ("C:\\Users\\jamie\\Desktop\\highDensitydragon.obj") };

    renderer.init();

    for (auto& m : meshes)
        renderer.add(m);

    renderer.render();

    auto fb = renderer.fbPtr();
    assert(fb);
    stbi_write_png(outFileName, fbSize.x, fbSize.y, 4, fb, fbSize.x * sizeof(uint32_t));

    renderer.release();
}
