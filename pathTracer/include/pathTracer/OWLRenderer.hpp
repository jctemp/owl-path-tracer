#ifndef BA_OWLRENDERER_HPP
#define BA_OWLRENDERER_HPP

#include <device/PathTracer.hpp>
#include <pathTracer/Renderer.hpp>

#include <vector>

owl::vec2i const fbSize{ 1920, 1080 };

extern "C" char PathTracer_ptx[];

namespace ba
{
	struct OWLRenderer : public Renderer
	{
	public:
		int init() override
		{
			context = owlContextCreate(nullptr, 1);
			module = owlModuleCreate(context, PathTracer_ptx);

			frameBuffer =
				owlHostPinnedBufferCreate(context, OWL_INT, fbSize.x * fbSize.y);

			OWLVarDecl missProgVars[]
			{
				{ "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color0)},
				{ "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color1)},
				{ nullptr }
			};

			missProg =
				owlMissProgCreate(context, module, "miss", sizeof(MissProgData), missProgVars, -1);

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

			rayGen =
				owlRayGenCreate(context, module, "simpleRayGen", sizeof(RayGenData), rayGenVars, -1);

			OWLVarDecl trianglesGeomVars[]
			{
				{ "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, index) },
				{ "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, vertex) },
				{ "color",  OWL_FLOAT3, OWL_OFFSETOF(TrianglesGeomData, color) },
				{ nullptr }
			};

			trianglesGeomType = owlGeomTypeCreate(context, OWLGeomKind::OWL_GEOM_TRIANGLES,
				sizeof(TrianglesGeomData), trianglesGeomVars, -1);

			owlGeomTypeSetClosestHit(trianglesGeomType, 0, module, "TriangleMesh");

			owlBuildPrograms(context);

			return 0;
		}

		int add(Mesh* m) override
		{
			ba::Mesh& mesh{ *m };

			auto& vertices{ mesh.vertex };
			auto& indices{ mesh.index };

			// set geometry in the buffers of the object
			OWLBuffer vertexBuffer{
				owlDeviceBufferCreate(context, OWL_FLOAT3, vertices.size(), vertices.data()) };

			OWLBuffer indexBuffer{
				owlDeviceBufferCreate(context, OWL_INT3, indices.size(), indices.data()) };

			// prepare mesh for device
			OWLGeom geom{
				owlGeomCreate(context, trianglesGeomType) };

			// set specific vertex/index buffer => required for build the accel.
			owlTrianglesSetVertices(geom, vertexBuffer,
				vertices.size(), sizeof(owl::vec3f), 0);

			owlTrianglesSetIndices(geom, indexBuffer,
				indices.size(), sizeof(owl::vec3i), 0);

			// set sbt data
			owlGeomSetBuffer(geom, "vertex", vertexBuffer);
			owlGeomSetBuffer(geom, "index", indexBuffer);
			owlGeomSet3f(geom, "color", owl3f{ 0.5f, 1.0f, 0 });

			geoms.push_back(geom);
			requireBuild = true;

			return 0;
		}

		int release() override
		{
			owlContextDestroy(context);
			return 0;
		}

		int renderSetting() override
		{
			std::runtime_error{ "int renderSetting() IS NOT IMPLEMENTED!" };
			return 0;
		}

		int render(Camera const& cam) override
		{
			if (geoms.size() > 0)
			{
				// Create Geom group and build world
				auto trianglesGroup =
					owlTrianglesGeomGroupCreate(context, geoms.size(), geoms.data());
				owlGroupBuildAccel(trianglesGroup);

				// Create an Instance group to make world
				world = owlInstanceGroupCreate(context, 1, &trianglesGroup);
				owlGroupBuildAccel(world);
			}
			else
			{
				world = owlInstanceGroupCreate(context, 0, nullptr);
				owlGroupBuildAccel(world);
			}


			
			owlMissProgSet3f(missProg, "color0", owl3f{ .8f,0.f,0.f });
			owlMissProgSet3f(missProg, "color1", owl3f{ .8f,.8f,.8f });

			float aspect{ fbSize.x / float(fbSize.y) };
			glm::vec3 camera_pos{ cam.lookFrom };
			glm::vec3 camera_d00{ glm::normalize(cam.lookAt - cam.lookFrom) };
			glm::vec3 camera_ddu{ cam.cosFovy * aspect * glm::normalize(glm::cross(camera_d00, cam.lookUp)) };
			glm::vec3 camera_ddv{ cam.cosFovy * glm::normalize(glm::cross(camera_ddu, camera_d00)) };
			camera_d00 -= 0.5f * camera_ddu;
			camera_d00 -= 0.5f * camera_ddv;

			owlRayGenSetBuffer(rayGen, "fbPtr", frameBuffer);
			owlRayGenSet2i(rayGen, "fbSize", (const owl2i&)fbSize);
			owlRayGenSetGroup(rayGen, "world", world);
			owlRayGenSet3f(rayGen, "camera.pos", (const owl3f&)camera_pos);
			owlRayGenSet3f(rayGen, "camera.dir_00", (const owl3f&)camera_d00);
			owlRayGenSet3f(rayGen, "camera.dir_du", (const owl3f&)camera_ddu);
			owlRayGenSet3f(rayGen, "camera.dir_dv", (const owl3f&)camera_ddv);

			owlBuildPrograms(context);
			owlBuildPipeline(context);
			owlBuildSBT(context);

			owlRayGenLaunch2D(rayGen, fbSize.x, fbSize.y);
			return 0;
		}

		uint32_t const* fbPtr() const override
		{
			return (const uint32_t*)owlBufferGetPointer(frameBuffer, 0);
		}

	public:
		// make context and shader
		OWLContext context;
		OWLModule module;

		// Programs
		OWLRayGen rayGen;
		OWLMissProg missProg;

		// link between host and device
		OWLBuffer frameBuffer;

		// Geometry and mesh
		OWLGeomType trianglesGeomType;
		std::vector<OWLGeom> geoms;
		bool requireBuild;

		// Group to handle geometry
		OWLGroup world;
	};
}

#endif // !BA_OWLRENDERER_HPP
