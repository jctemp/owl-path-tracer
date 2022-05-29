#include "types.hpp"
#include "device.hpp"
#include "path_tracer.hpp"

__constant__ launch_params_data optixLaunchParams;

inline __device__ vec3 pow(vec3 const& v, float const& p)
{
    return vec3{powf(v.x, p), powf(v.y, p), powf(v.z, p)};
}

__device__ void load_triangle_indices(int32_t const& mesh_id, int32_t const& primitive_id, ivec3& indices)
{
    get_data(auto indices_buffer, optixLaunchParams.indices_buffer, mesh_id, Buffer);
    get_data(indices, indices_buffer, primitive_id, ivec3);
}

__device__ void load_triangle_vertices(int32_t const& mesh_id, ivec3 const& indices, vec2 const& barycentric,
                                       vec3& position, vec3& geometric_normal)
{
    get_data(auto vertices_buffer, optixLaunchParams.vertices_buffer, mesh_id, Buffer);
    get_data(auto p0, vertices_buffer, indices.x, vec3);
    get_data(auto p1, vertices_buffer, indices.y, vec3);
    get_data(auto p2, vertices_buffer, indices.z, vec3);

    position = (1 - barycentric.x - barycentric.y) * p0 + barycentric.x * p1 + barycentric.y * p2;
    geometric_normal = normalize(cross(p1 - p0, p2 - p0));
}

__device__ void load_triangle_normals(int32_t const& mesh_id, ivec3 const& indices, vec2 const& barycentric,
                                      vec3& shading_normal)
{
    get_data(auto normals_buffer, optixLaunchParams.normals_buffer, mesh_id, Buffer);
    get_data(auto n0, normals_buffer, indices.x, vec3);
    get_data(auto n1, normals_buffer, indices.y, vec3);
    get_data(auto n2, normals_buffer, indices.z, vec3);

    shading_normal = normalize((1 - barycentric.x - barycentric.y) * n0 + barycentric.x * n1 + barycentric.y * n2);
}

OPTIX_RAYGEN_PROGRAM(ray_gen)()
{
    ray_gen_data const& self{owl::getProgramData<ray_gen_data>()};
    ivec2 const pixelId{owl::getLaunchIndex()};
    random pxRand{(uint32_t) pixelId.x, (uint32_t) pixelId.y};

    vec3 color{0.0f};
    for (int32_t s{0}; s < optixLaunchParams.max_samples; ++s)
    {
        // shot ray with slight randomness to make soft edges
        vec2 const rand{pxRand(), pxRand()};
        vec2 const screen{(vec2{pixelId} + rand) / vec2{self.fb_size}};

        // determine initial ray form the camera
        radiance_ray ray{
                self.camera.origin,
                normalize(
                        self.camera.llc + screen.u * self.camera.horizontal + screen.v * self.camera.vertical -
                        self.camera.origin),
                t_min, t_max};

        color += trace_path(ray, pxRand, s);
    }

    // take the average of all samples per pixel and apply gamma correction
    color *= 1.0f / static_cast<float>(optixLaunchParams.max_samples);
    color = o_saturate(pow(color, 1.0f / 2.2f));

    assert_condition(isinf(color.x) || isinf(color.y) || isinf(color.z), "inf detected\n")
    assert_condition(isnan(color.x) || isnan(color.y) || isnan(color.z), "nan detected\n")

    // save result into the buffer
    const int fbOfs = pixelId.x + self.fb_size.x * (self.fb_size.y - 1 - pixelId.y);
    self.fb_ptr[fbOfs]
            = owl::make_rgba(color);
}

OPTIX_CLOSEST_HIT_PROGRAM(triangle_hit)()
{
    per_ray_data& prd{owl::getPRD<per_ray_data>()};

    prd.is->t = optixGetRayTmax();

    // barycentrics
    float b1{optixGetTriangleBarycentrics().x};
    float b2{optixGetTriangleBarycentrics().y};
    float b0{1 - b1 - b2};

    prd.is->uv = {b1, b2};

    // get direction
    auto ray_dir{optixGetWorldRayDirection()};
    vec3 const direction{
            ray_dir.x,
            ray_dir.y,
            ray_dir.z
    };

    prd.is->wo = -direction;

    // get geometric data:
    auto const& self = owl::getProgramData<triangle_geom_data>();
    uint32_t const primID{optixGetPrimitiveIndex()};
    auto const mesh_id{self.id};

    ivec3 index{};
    load_triangle_indices(mesh_id, primID, index);
    load_triangle_vertices(mesh_id, index, prd.is->uv, prd.is->position, prd.is->normal_geometric);
    load_triangle_normals(mesh_id, index, prd.is->uv, prd.is->normal);

    prd.is->material_id = self.material_id;
    prd.is->prim = primID;

    // scatter event type
    prd.scatter_event = scatter_event::bounced;
}

OPTIX_MISS_PROGRAM(miss)()
{
    per_ray_data& prd{owl::getPRD<per_ray_data>()};
    prd.scatter_event = scatter_event::missed;
}

OPTIX_MISS_PROGRAM(miss_shadow)()
{
    bool& prd{owl::getPRD<bool>()};
    prd = true;
}