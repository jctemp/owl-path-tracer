#include <owl/owl_device.h>

#include "types.hpp"
#include "device.hpp"
#include "materials.hpp"
#include "macros.hpp"

using radiance_ray = owl::RayT<0, 2>;
using shadow_ray = owl::RayT<1, 2>;

/// this constant must be called optixLaunchParams<br>
/// it is declared in path_tracer extern/owl/owl/DeviceContext.cpp
__constant__ launch_params_data optixLaunchParams;


inline __device__ vec3 pow(vec3 const& v, float const& p)
{
    return vec3{powf(v.x, p), powf(v.y, p), powf(v.z, p)};
}


inline __device__ vec2 uv_on_sphere(vec3 n)
{
    auto const u{0.5f + atan2(n.x, n.z) / (2.0f * pi)};
    auto const v{0.5f + asin(n.y) / pi};
    return vec2{u, v};
}


inline __device__ vec3 sample_environment(vec3 dir)
{
    auto& launch_params {optixLaunchParams};

    vec2 tc{uv_on_sphere(dir)};
    owl::vec4f const texColor{
            tex2D<float4>(launch_params.environment_map, tc.x, tc.y)};
    return vec3{texColor};
}


inline __device__ void load_triangle_indices(int32_t const& mesh_index, int32_t const& primitive_id,
                                             ivec3& indices)
{
    auto& launch_params = optixLaunchParams;
    get_data(auto indices_buffer, launch_params.indices_buffer, mesh_index, Buffer);
    get_data(indices, indices_buffer, primitive_id, ivec3);
}

inline __device__ void load_triangle_vertices(int32_t const& mesh_index, ivec3 const& indices, vec2 const& barycentric,
                                       vec3& position, vec3& geometric_normal)
{
    auto& launch_params = optixLaunchParams;
    get_data(auto vertices_buffer, launch_params.vertices_buffer, mesh_index, Buffer);
    get_data(auto p0, vertices_buffer, indices.x, vec3);
    get_data(auto p1, vertices_buffer, indices.y, vec3);
    get_data(auto p2, vertices_buffer, indices.z, vec3);

    position = (1 - barycentric.x - barycentric.y) * p0 + barycentric.x * p1 + barycentric.y * p2;
    geometric_normal = normalize(cross(p1 - p0, p2 - p0));
}

inline __device__ void load_triangle_normals(int32_t const& mesh_index, ivec3 const& indices, vec2 const& barycentric,
                                      vec3& shading_normal)
{
    auto& launch_params = optixLaunchParams;
    get_data(auto normals_buffer, launch_params.normals_buffer, mesh_index, Buffer);
    get_data(auto n0, normals_buffer, indices.x, vec3);
    get_data(auto n1, normals_buffer, indices.y, vec3);
    get_data(auto n2, normals_buffer, indices.z, vec3);

    shading_normal = normalize((1 - barycentric.x - barycentric.y) * n0 + barycentric.x * n1 + barycentric.y * n2);
}


inline __device__ bool visibiliy_test(vec3 const& position, vec3 const& direction, float const& max_distance)
{
    auto& launch_params {optixLaunchParams};

    // check the space between position and target only
    shadow_ray light_ray{position, direction, t_min, max_distance - t_min};
    bool visible{false};

    owl::traceRay(launch_params.world, light_ray, visible,
              OPTIX_RAY_FLAG_DISABLE_ANYHIT
            | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);

    return visible;
}


__device__ vec3 trace_path(radiance_ray& ray, random& random, int32_t& samples)
{
    auto& launch_params = optixLaunchParams;

    /// total accumulation of existent radiance
    vec3 radiance{0.0f};


    /// keeps tracks of the throughput of a ray, it is weighting the radiance <br>
    /// beta = f * cos(theta) / pdf
    vec3 beta{1.0f};

    hit_data hd;
    material_data ms;
    per_ray_data prd{random, scatter_event::none, &hd, &ms};
    material_type sampled_type{};

    for (int32_t depth{0}; depth < launch_params.max_path_depth; ++depth)
    {
        /// find closest intersection
        owl::traceRay(launch_params.world, ray, prd);

        /// miss then terminate the path and sample environment
        if (prd.scatter_event == scatter_event::miss)
        {
            if (launch_params.environment_use && launch_params.environment_map)
                radiance += sample_environment(ray.direction) * beta;
            else if (launch_params.environment_auto)
                radiance += lerp(vec3{1.0f}, vec3{0.5f, 0.7f, 1.0f}, 0.5f * (ray.direction.y + 1.0f)) * beta;
            else
                radiance += launch_params.environment_color * launch_params.environment_intensity * beta;
            break;
        }

        /// load mesh for interaction calculations
        ivec3 indices{};
        vec3 v_p{}, v_gn{}, v_n{};

        load_triangle_indices(hd.mesh_index, hd.primitive_index, indices);
        load_triangle_vertices(hd.mesh_index, indices, hd.barycentric, v_p, v_gn);
        load_triangle_normals(hd.mesh_index, indices, hd.barycentric, v_n);

        vec3 wo{hd.wo}, wi{};

        vec3 T{}, B{};
        onb(v_n, T, B);

        float pdf{};
        vec3 f{};

        material_data material{};
        if (hd.material_index >= 0)
        {
            get_data(material, launch_params.material_buffer, hd.material_index, material_data);
        }

        vec3 local_wo{to_local(T, B, v_n, wo)}, local_wi{}, local_wh{};

        sample_lambert(material, local_wo, prd.random, local_wi, f, pdf, sampled_type);

        //sample_disney_bsdf(material, local_wo, prd.random,
        //        local_wi, local_wh, f, pdf, sampled_type);

        wi = to_world(T, B, v_n, local_wi);

        /// terminate or catching de-generate paths
        if (pdf < 1E-5f)
            break;

        if (has_inf(f) || has_nan(f))
        {
            --depth; // invalid path and re-sample
            continue;
        }

        beta *= (f * owl::abs(owl::dot(wi, v_n))) / pdf;

        /// terminate path by random
        auto const beta_max{owl::max(beta.x, owl::max(beta.y, beta.z))};
        if (depth > 3)
        {
            float q{owl::max(.05f, 1 - beta_max)};
            if (prd.random() > q) break;
        }

        ray = radiance_ray{v_p, wi, t_min, t_max};
    }

    return radiance;
}

OPTIX_RAYGEN_PROGRAM(ray_gen)()
{
    auto& launch_params = optixLaunchParams;

    ray_gen_data const& self{owl::getProgramData<ray_gen_data>()};
    ivec2 const pixelId{owl::getLaunchIndex()};
    random pxRand{(uint32_t) pixelId.x, (uint32_t) pixelId.y};

    vec3 color{0.0f};
    for (int32_t s{0}; s < launch_params.max_samples; ++s)
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
    color *= 1.0f / static_cast<float>(launch_params.max_samples);
    color = o_saturate(pow(color, 1.0f / 2.2f));

    // save result into the buffer
    const int fbOfs = pixelId.x + self.fb_size.x * (self.fb_size.y - 1 - pixelId.y);
    self.fb_ptr[fbOfs]
            = owl::make_rgba(color);
}

OPTIX_CLOSEST_HIT_PROGRAM(triangle_hit)()
{
    per_ray_data& prd{owl::getPRD<per_ray_data>()};

    auto const& barycentric{optixGetTriangleBarycentrics()};
    auto const& ray_t_max{optixGetRayTmax()};
    auto const& ray_direction{optixGetWorldRayDirection()};
    auto const& primitive_index{optixGetPrimitiveIndex()};

    prd.hd->barycentric = barycentric;
    prd.hd->t = ray_t_max;
    prd.hd->wo = ray_direction;
    prd.hd->wo = -normalize(prd.hd->wo);
    prd.hd->primitive_index = primitive_index;

    auto const& self = owl::getProgramData<entity_data>();

    prd.hd->material_index = self.material_index;
    prd.hd->mesh_index = self.mesh_index;

    prd.scatter_event = scatter_event::hit;
}

OPTIX_MISS_PROGRAM(miss)()
{
    per_ray_data& prd{owl::getPRD<per_ray_data>()};
    prd.scatter_event = scatter_event::miss;
}

OPTIX_MISS_PROGRAM(miss_shadow)()
{
    bool& prd{owl::getPRD<bool>()};
    prd = true;
}