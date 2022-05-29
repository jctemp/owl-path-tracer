
#ifndef PATH_TRACER_LIGHTS_HPP
#define PATH_TRACER_LIGHTS_HPP

#include "types.hpp"
#include "macros.hpp"

/*
 * AREA LIGHTS
 * - power: Lemit * area * PI
 * - pdf: 1 / (area)
 */

float triangle_area(vec3 const& a, vec3 const& b, vec3 const& c)
{
    auto const ab{b - a};
    auto const ac{c - a};
    return 0.5f * length(cross(ab, ac));
}

interface_data triangle_sample(vec3 const& a, vec3 const& b, vec3 const& c,
                               random& random, float& pdf)
{
    auto const u{random()};
    auto const v{random()};
    auto const w{1.0f - u - v};
    auto const barycentric{vec3{u, v, w}};
    auto const p{a * barycentric.x + b * barycentric.y + c * barycentric.z};
    auto const n{normalize(cross(b - a, c - a))};
    auto const area{triangle_area(a, b, c)};
    if (area == 0.0f) pdf = 0.0f;
    else pdf = 1.0f / area;

    interface_data is{};
    is.triangle_points[0] = a;
    is.triangle_points[1] = b;
    is.triangle_points[2] = c;
    is.position = p;
    is.normal = n;
    is.geometric_normal = n;
    is.wo = -n;
    is.uv = {u, v};
    is.t = 0.0f;
    is.prim = 0;
    is.type = 0;
    is.id = 0;

    return is;
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
