#include <device/PathTracer.hpp>
#include <pathTracer/Renderer.hpp>
#include <pathTracer/ObjLoader.hpp>
#include <pathTracer/StbUtils.hpp>
#include <pathTracer/OWLRenderer.hpp>

#include <map>
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

    ba::ImageRgb environmentTexture{};
    ba::loadImage(environmentTexture, "rooitou_park_4k.hdr", "C:/Users/jamie/Desktop");

    renderer.init();

    renderer.setEnvironmentTexture(environmentTexture);

    for (auto& m : meshes)
        renderer.add(m);

    renderer.render(cam);

    ba::Image result{ fbSize.x, fbSize.y, renderer.fbPtr() };
    ba::writeImage(result, "image.png");

    renderer.release();
}
