
#ifndef PATH_TRACER_LIGHTS_HPP
#define PATH_TRACER_LIGHTS_HPP

#include "types.hpp"
#include "macros.hpp"
#include "sample_methods.hpp"

/*
 * AREA LIGHTS
 * - power: Lemit * area * PI
 * - pdf: 1 / (area)
 */

__both__ float triangle_area(vec3 const& a, vec3 const& b, vec3 const& c)
{
    auto const ab{b - a};
    auto const ac{c - a};
    return 0.5f * length(cross(ab, ac));
}


__both__ float pdf_area_to_solid_angle(float pdf_area, float dist_sqr_area, float cos_theta)
{
    float abs_cos_theta = abs(cos_theta);
    if( abs_cos_theta < 1e-4f) return 0.0;
    return pdf_area * dist_sqr_area / abs_cos_theta;
}

__both__ void sample_triangle(vec3 const& p0, vec3 const& p1, vec3 const& p2,
                              vec3 const& n0, vec3 const& n1, vec3 const& n2,
                              vec3 const& target, random& random,
                              vec3& direction, float& distance, float& pdf, vec2& barycentric,
                              vec3& normal)
{
    barycentric = {uniform_sample_triangle({random(), random()})};
    auto position{(1.0f - barycentric.x - barycentric.y) * p0 + barycentric.x * p1 + barycentric.y * p2};
    normal = (1.0f - barycentric.x - barycentric.y) * n0 + barycentric.x * n1 + barycentric.y * n2;

    auto area{triangle_area(p0, p1, p2)};
    direction = {position - target};
    auto distance_sqr{dot(direction, direction)};
    distance = {owl::sqrt(distance_sqr)};
    direction /= distance;

    auto cos_theta{dot(-direction, normal)};
    pdf = pdf_area_to_solid_angle(1.0f / area, distance_sqr, cos_theta);
}

__both__ float pdf_a_to_w(float pdf_a, float dist, float theta) {
    auto abs_cos_theta{ owl::abs(theta) };
	if (abs_cos_theta < 1e-4) return 0.f;
	return pdf_a * sqr(dist) / abs_cos_theta;
}

__both__ float power_heuristic(float n_f, float pdf_f, float n_g, float pdf_g)
{
    auto f{ n_f * pdf_f };
    auto g{ n_g * pdf_g };
    return (f * f) / (f * f + g * g);
}

#endif //PATH_TRACER_LIGHTS_HPP
