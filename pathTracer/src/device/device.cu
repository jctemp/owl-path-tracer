#include "types.hpp"
#include "device.hpp"
#include "path_tracer.hpp"

__constant__ launch_params_data optixLaunchParams;

OPTIX_RAYGEN_PROGRAM(ray_gen)()
{
	ray_gen_data const& self{ owl::getProgramData<ray_gen_data>() };
	ivec2 const pixelId{ owl::getLaunchIndex() };
	Random pxRand{ (uint32_t)pixelId.x, (uint32_t)pixelId.y };

	vec3 color{ 0.0f };
	for (int32_t s{ 0 }; s < optixLaunchParams.max_samples; ++s)
	{
		// shot ray with slight randomness to make soft edges
		vec2 const rand{ pxRand(), pxRand() };
		vec2 const screen{ (vec2{pixelId} + rand) / vec2{self.fbSize} };

		// determine initial ray form the camera
		radiance_ray ray{
			self.camera.origin,
			normalize(
				self.camera.llc + screen.u * self.camera.horizontal + screen.v * self.camera.vertical - self.camera.origin),
            t_min, t_max };

		color += trace_path(ray, pxRand);
	}

	// take the average of all samples per pixel and apply gamma correction
	color *= 1.0f / static_cast<float>(optixLaunchParams.max_samples);
	color = owl::sqrt(color);
	color = o_saturate(color);

    assert_condition(isinf(color.x) || isinf(color.y) || isinf(color.z), "inf detected\n")
    assert_condition(isnan(color.x) || isnan(color.y) || isnan(color.z), "nan detected\n")

	// save result into the buffer
	const int fbOfs = pixelId.x + self.fbSize.x * (self.fbSize.y - 1 - pixelId.y);
	self.fbPtr[fbOfs]
		= owl::make_rgba(color);
}

OPTIX_CLOSEST_HIT_PROGRAM(triangle_hit)()
{
    per_ray_data& prd{ owl::getPRD<per_ray_data>() };

    prd.is->t = optixGetRayTmax();

    // barycentrics
    float b1{ optixGetTriangleBarycentrics().x };
    float b2{ optixGetTriangleBarycentrics().y };
    float b0{ 1 - b1 - b2 };

    prd.is->UV = { b1, b2 };

    // get direction
    auto ray_dir{ optixGetWorldRayDirection() };
    vec3 const direction{
            ray_dir.x,
            ray_dir.y,
            ray_dir.z
    };

    prd.is->V = -direction;

    // get geometric data:
    triangle_geom_data const& self = owl::getProgramData<triangle_geom_data>();
    uint32_t const primID{ optixGetPrimitiveIndex() };
    ivec3 const index{ self.index[primID] };

    prd.is->matId = self.matId;
    prd.is->lightId = self.lightId;
    prd.is->prim = primID;

    // vertices for P and Ng
    vec3 const& p0{ self.vertex[index.x] };
    vec3 const& p1{ self.vertex[index.y] };
    vec3 const& p2{ self.vertex[index.z] };

    prd.is->TRI[0] = vec3{ p0 };
    prd.is->TRI[1] = vec3{ p1 };
    prd.is->TRI[2] = vec3{ p2 };

    prd.is->Ng = normalize(cross(p1 - p0, p2 - p0));
    prd.is->P = p0 * b0 + p1 * b1 + p2 * b2;

    // vertex normals for N
    vec3 const& n0{ self.normal[index.x] };
    vec3 const& n1{ self.normal[index.y] };
    vec3 const& n2{ self.normal[index.z] };

    prd.is->N = normalize(n0 * b0 + n1 * b1 + n2 * b2);

    // scatter event type
    prd.scatterEvent = scatter_event::BOUNCED;
}

OPTIX_MISS_PROGRAM(miss)()
{
    per_ray_data& prd{ owl::getPRD<per_ray_data>() };
    prd.scatterEvent = scatter_event::MISS;
}

OPTIX_MISS_PROGRAM(miss_shadow)()
{
    bool& prd{ owl::getPRD<bool>() };
    prd = true;
}