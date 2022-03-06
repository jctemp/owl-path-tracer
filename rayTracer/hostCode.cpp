#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "deviceCode.hpp"
#include "objLoader.hpp"

#include <stb_image_write.h>
#include <simpleLogger.hpp>

using namespace owl;

const int NUM_VERTICES = 8;
vec3f vertices[NUM_VERTICES]
{
  { -1.f,-1.f,-1.f },
  { +1.f,-1.f,-1.f },
  { -1.f,+1.f,-1.f },
  { +1.f,+1.f,-1.f },
  { -1.f,-1.f,+1.f },
  { +1.f,-1.f,+1.f },
  { -1.f,+1.f,+1.f },
  { +1.f,+1.f,+1.f }
};

const int NUM_INDICES = 12;
vec3i indices[NUM_INDICES]
{
  { 0,1,3 }, { 2,3,0 },
  { 5,7,6 }, { 5,6,4 },
  { 0,4,5 }, { 0,5,1 },
  { 2,3,7 }, { 2,7,6 },
  { 1,5,7 }, { 1,7,3 },
  { 4,0,2 }, { 4,2,6 }
};

char const* outFileName{ "image.png" };
vec2i const fbSize{ 800, 600 };
vec3f const lookFrom(4.f, 0.f, 0.f);
vec3f const lookAt(0.f, 0.f, 0.f);
vec3f const lookUp(0.f, 1.f, 0.f);
float const cosFovy = 0.66f;

// ------------------------------------------------------------------

extern "C" char deviceCode_ptx[];

extern "C" int main(int argc, char *argv[])
{
    SL_LOG("Loading OBJ model");

    std::vector<ba::TrianglesMesh*> meshes{ ba::loadOBJ("C:\\Users\\jamie\\Desktop\\Sphere.obj") };

    auto vertices = meshes[0]->vertex;
    auto indices = meshes[0]->index;

    SL_OK("Loaded successfully model");


    SL_LOG(fmt::format("Starting program {}", argv[0]));

    OWLContext context = owlContextCreate(nullptr, 1);
    OWLModule module = owlModuleCreate(context, deviceCode_ptx);

    SL_LOG("Creating triangles variable");

    OWLVarDecl trianglesGeomVars[]
    {
      { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, index) },
      { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, vertex) },
      { "color",  OWL_FLOAT3, OWL_OFFSETOF(TrianglesGeomData, color) }
    };

    OWLGeomType trianglesGeomType{owlGeomTypeCreate(
        context, OWLGeomKind::OWL_GEOM_TRIANGLES, 
        sizeof(TrianglesGeomData),trianglesGeomVars, 3)
    };

    owlGeomTypeSetClosestHit(trianglesGeomType, 0, module, "TriangleMesh");

    SL_LOG("Building geometries...");

    // set geometry in the buffers of the object
    OWLBuffer vertexBuffer{ 
        owlDeviceBufferCreate(context, OWL_FLOAT3, vertices.size(), vertices.data())};
    OWLBuffer indexBuffer{ 
        owlDeviceBufferCreate(context, OWL_INT3, indices.size(), indices.data())};
    OWLBuffer frameBuffer{ 
        owlHostPinnedBufferCreate(context, OWL_INT, fbSize.x * fbSize.y) };

    // create Geom for group
    OWLGeom trianglesGeom{
        owlGeomCreate(context, trianglesGeomType)};

    // associate buffers with the trianglesGeom
    owlTrianglesSetVertices(trianglesGeom, vertexBuffer, 
        vertices.size(), sizeof(vec3f), 0);
    owlTrianglesSetIndices(trianglesGeom, indexBuffer,
        indices.size(), sizeof(vec3i), 0);

    owlGeomSetBuffer(trianglesGeom, "vertex", vertexBuffer);
    owlGeomSetBuffer(trianglesGeom, "index", indexBuffer);
    owlGeomSet3f(trianglesGeom, "color", owl3f{ 0.5f, 1.0f, 0 });

    // Create Geom group and build world
    OWLGroup trianglesGroup{
        owlTrianglesGeomGroupCreate(context, 1, &trianglesGeom) };
    owlGroupBuildAccel(trianglesGroup);

    // Create an Instance group to make world
    OWLGroup world{ owlInstanceGroupCreate(context, 1, &trianglesGroup) };
    owlGroupBuildAccel(world);

    SL_LOG("Setup MissProg");

    OWLVarDecl missProgVars[]
    {
        { "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color0)},
        { "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color1)},
        { nullptr }
    };

    OWLMissProg missProg{
        owlMissProgCreate(context, module, "miss", sizeof(MissProgData), missProgVars, -1) };

    owlMissProgSet3f(missProg, "color0", owl3f{ .8f,0.f,0.f });
    owlMissProgSet3f(missProg, "color1", owl3f{ .8f,.8f,.8f });

    SL_LOG("Setup RayGenProg");

    OWLVarDecl rayGenVars[]
    {
      { "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData, fbPtr)},
      { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData, fbSize)},
      { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData, world)},
      { "camera.pos",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.pos)},
      { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_00)},
      { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_du)},
      { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_dv)},
      { nullptr }
    };

    OWLRayGen rayGen{ 
        owlRayGenCreate(context, module, "simpleRayGen", sizeof(RayGenData), rayGenVars, -1) };

    float aspect{ fbSize.x / float(fbSize.y) };
    vec3f camera_pos{ lookFrom };
    vec3f camera_d00{ normalize(lookAt - lookFrom) };
    vec3f camera_ddu{ cosFovy * aspect * normalize(cross(camera_d00, lookUp)) };
    vec3f camera_ddv{ cosFovy * normalize(cross(camera_ddu, camera_d00)) };
    camera_d00 -= 0.5f * camera_ddu;
    camera_d00 -= 0.5f * camera_ddv;

    owlRayGenSetBuffer(rayGen, "fbPtr", frameBuffer);
    owlRayGenSet2i(rayGen, "fbSize", (const owl2i&)fbSize);
    owlRayGenSetGroup(rayGen, "world", world);
    owlRayGenSet3f(rayGen, "camera.pos", (const owl3f&)camera_pos);
    owlRayGenSet3f(rayGen, "camera.dir_00", (const owl3f&)camera_d00);
    owlRayGenSet3f(rayGen, "camera.dir_du", (const owl3f&)camera_ddu);
    owlRayGenSet3f(rayGen, "camera.dir_dv", (const owl3f&)camera_ddv);

    SL_LOG("Build program, pipeline, sbt");

    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);

    SL_LOG("Launching ...");

    owlRayGenLaunch2D(rayGen, fbSize.x, fbSize.y);

    SL_LOG("done with launch, writing picture ...");

    // for host pinned mem it doesn't matter which device we query...
    const uint32_t* fb{ (const uint32_t*)owlBufferGetPointer(frameBuffer, 0) };
    assert(fb);
    stbi_write_png(outFileName, fbSize.x, fbSize.y, 4, fb, fbSize.x * sizeof(uint32_t));

    SL_OK(fmt::format("written rendered frame buffer to file {}", outFileName));

    SL_LOG("destroying devicegroup ...");
    owlContextDestroy(context);

    SL_OK("seems all went OK; app is done, this should be the last output ...");
}
