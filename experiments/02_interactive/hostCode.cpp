#include <owl/owl.h>
#include <owlViewer/OWLViewer.h>
#include <simpleLogger.hpp>

#include "device/tracer.hpp"

using namespace owl;

extern "C" char tracer_ptx[];

const vec3f init_lookFrom(0.0f, 0.0f, 5.0f);
const vec3f init_lookAt(0.0f, 0.0f, -1.0f);
const vec3f init_lookUp(0.0f, 1.0f, 0.0f);
const float init_cosFovy{ 0.66f };

const int NUM_VERTICES{ 8 };
vec3f vertices[]
{	// x y z
	{ -1.f,-1.f,-1.f },
	{ +1.f,-1.f,-1.f },
	{ -1.f,+1.f,-1.f },
	{ +1.f,+1.f,-1.f },
	{ -1.f,-1.f,+1.f },
	{ +1.f,-1.f,+1.f },
	{ -1.f,+1.f,+1.f },
	{ +1.f,+1.f,+1.f }
};

const int NUM_INDICES{ 12 };
vec3i indices[]
{
	{ 0,1,3 }, { 2,3,0 },
	{ 5,7,6 }, { 5,6,4 },
	{ 0,4,5 }, { 0,5,1 },
	{ 2,3,7 }, { 2,7,6 },
	{ 1,5,7 }, { 1,7,3 },
	{ 4,0,2 }, { 4,2,6 }
};

struct Viewer : public owl::viewer::OWLViewer
{
    Viewer();

    /*! gets called whenever the viewer needs us to re-render out widget */
    void render() override;

    /*! window notifies us that we got resized. We HAVE to override
        this to know our actual render dimensions, and get pointer
        to the device frame buffer that the viewer cated for us */
    void resize(const vec2i& newSize) override;

    /*! this function gets called whenever any camera manipulator
      updates the camera. gets called AFTER all values have been updated */
    void cameraChanged() override;

    bool sbtDirty = true;
    OWLRayGen rayGen{ 0 };
    OWLContext context{ 0 };
};

/*! window notifies us that we got resized */
void Viewer::resize(const vec2i& newSize)
{
    OWLViewer::resize(newSize);
    cameraChanged();
}

/*! window notifies us that the camera has changed */
void Viewer::cameraChanged()
{
    const vec3f lookFrom = camera.getFrom();
    const vec3f lookAt = camera.getAt();
    const vec3f lookUp = camera.getUp();
    const float cosFovy = camera.getCosFovy();
    // ----------- compute variable values  ------------------
    vec3f camera_pos = lookFrom;
    vec3f camera_d00
        = normalize(lookAt - lookFrom);
    float aspect = fbSize.x / float(fbSize.y);
    vec3f camera_ddu
        = cosFovy * aspect * normalize(cross(camera_d00, lookUp));
    vec3f camera_ddv
        = cosFovy * normalize(cross(camera_ddu, camera_d00));
    camera_d00 -= 0.5f * camera_ddu;
    camera_d00 -= 0.5f * camera_ddv;

    // ----------- set variables  ----------------------------
    owlRayGenSet1ul(rayGen, "fbPtr", (uint64_t)fbPointer);
    // owlRayGenSetBuffer(rayGen,"fbPtr",        frameBuffer);
    owlRayGenSet2i(rayGen, "fbSize", (const owl2i&)fbSize);
    owlRayGenSet3f(rayGen, "camera.pos", (const owl3f&)camera_pos);
    owlRayGenSet3f(rayGen, "camera.dir_00", (const owl3f&)camera_d00);
    owlRayGenSet3f(rayGen, "camera.dir_du", (const owl3f&)camera_ddu);
    owlRayGenSet3f(rayGen, "camera.dir_dv", (const owl3f&)camera_ddv);
    sbtDirty = true;
}

Viewer::Viewer()
{
    // create a context on the first device:
    context = owlContextCreate(nullptr, 1);
    OWLModule module = owlModuleCreate(context, tracer_ptx);

    // ##################################################################
    // set up all the *GEOMETRY* graph we want to render
    // ##################################################################

    // -------------------------------------------------------
    // declare geometry type
    // -------------------------------------------------------
    OWLVarDecl trianglesGeomVars[] = {
      { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
      { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
      { "color",  OWL_FLOAT3, OWL_OFFSETOF(TrianglesGeomData,color)}
    };
    OWLGeomType trianglesGeomType
        = owlGeomTypeCreate(context,
            OWL_TRIANGLES,
            sizeof(TrianglesGeomData),
            trianglesGeomVars, 3);
    owlGeomTypeSetClosestHit(trianglesGeomType, 0,
        module, "TriangleMesh");

    // ##################################################################
    // set up all the *GEOMS* we want to run that code on
    // ##################################################################

    LOG("building geometries ...");

    // ------------------------------------------------------------------
    // triangle mesh
    // ------------------------------------------------------------------
    OWLBuffer vertexBuffer
        = owlDeviceBufferCreate(context, OWL_FLOAT3, NUM_VERTICES, vertices);
    OWLBuffer indexBuffer
        = owlDeviceBufferCreate(context, OWL_INT3, NUM_INDICES, indices);
    // OWLBuffer frameBuffer
    //   = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x*fbSize.y);

    OWLGeom trianglesGeom
        = owlGeomCreate(context, trianglesGeomType);

    owlTrianglesSetVertices(trianglesGeom, vertexBuffer,
        NUM_VERTICES, sizeof(vec3f), 0);
    owlTrianglesSetIndices(trianglesGeom, indexBuffer,
        NUM_INDICES, sizeof(vec3i), 0);

    owlGeomSetBuffer(trianglesGeom, "vertex", vertexBuffer);
    owlGeomSetBuffer(trianglesGeom, "index", indexBuffer);
    owlGeomSet3f(trianglesGeom, "color", owl3f{ 0,1,0 });

    // ------------------------------------------------------------------
    // the group/accel for that mesh
    // ------------------------------------------------------------------
    OWLGroup trianglesGroup
        = owlTrianglesGeomGroupCreate(context, 1, &trianglesGeom);
    owlGroupBuildAccel(trianglesGroup);
    OWLGroup world
        = owlInstanceGroupCreate(context, 1, &trianglesGroup);
    owlGroupBuildAccel(world);


    // ##################################################################
    // set miss and raygen program required for SBT
    // ##################################################################

    // -------------------------------------------------------
    // set up miss prog
    // -------------------------------------------------------
    OWLVarDecl missProgVars[]
        = {
        { "color0", OWL_FLOAT3, OWL_OFFSETOF(MissData,color0)},
        { "color1", OWL_FLOAT3, OWL_OFFSETOF(MissData,color1)},
        { /* sentinel to mark end of list */ }
    };
    // ----------- create object  ----------------------------
    OWLMissProg missProg
        = owlMissProgCreate(context, module, "miss", sizeof(MissData),
            missProgVars, -1);

    // ----------- set variables  ----------------------------
    owlMissProgSet3f(missProg, "color0", owl3f{ .8f,0.f,0.f });
    owlMissProgSet3f(missProg, "color1", owl3f{ .8f,.8f,.8f });

    // -------------------------------------------------------
    // set up ray gen program
    // -------------------------------------------------------
    OWLVarDecl rayGenVars[] = {
      { "fbPtr",         OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData,fbPtr)},
      // { "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr)},
      { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
      { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
      { "camera.pos",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.pos)},
      { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_00)},
      { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_du)},
      { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_dv)},
      { /* sentinel to mark end of list */ }
    };

    // ----------- create object  ----------------------------
    rayGen
        = owlRayGenCreate(context, module, "rayGen",
            sizeof(RayGenData),
            rayGenVars, -1);
    /* camera and frame buffer get set in resiez() and cameraChanged() */
    owlRayGenSetGroup(rayGen, "world", world);

    // ##################################################################
    // build *SBT* required to trace the groups
    // ##################################################################

    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);
}

void Viewer::render()
{
    if (sbtDirty) {
        owlBuildSBT(context);
        sbtDirty = false;
    }
    owlRayGenLaunch2D(rayGen, fbSize.x, fbSize.y);
}


int main(int ac, char** av)
{
    Viewer viewer;
    viewer.camera.setOrientation(init_lookFrom,
        init_lookAt,
        init_lookUp,
        owl::viewer::toDegrees(acosf(init_cosFovy)));
    viewer.enableFlyMode();
    viewer.enableInspectMode(owl::box3f(vec3f(-1.f), vec3f(+1.f)));

    viewer.showAndRun();
}