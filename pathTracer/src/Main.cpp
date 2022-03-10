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

    std::vector<ba::Mesh*> meshes{ ba::loadOBJ("C:\\Users\\jamie\\Desktop\\Dragon.obj") };

    ba::Camera cam{ 
        {2.0f,1.0f,0.0f}, // look from
        {0.0f,0.5f,0.0f}, // look at
        {0.0f,1.0f,0.0f}, // look up
        0.88f // cosFov
    }; 

    renderer.init();

    for (auto& m : meshes)
        renderer.add(m);

    renderer.render(cam);

    auto fb = renderer.fbPtr();
    assert(fb);
    stbi_write_png("image.png", fbSize.x, fbSize.y, 4, fb, fbSize.x * sizeof(uint32_t));

    renderer.release();
}
