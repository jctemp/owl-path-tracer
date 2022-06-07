#ifndef PATH_TRACER_DISNEY_HELPER_CUH
#define PATH_TRACER_DISNEY_HELPER_CUH

inline __both__ float luminance(const vec3& color)
{
    return owl::dot(vec3{0.2126f, 0.7152f, 0.0722f}, color);
}

inline __both__ float clamp(float x, float low, float high)
{
    return fmin(high, fmax(low, x));
}

inline __both__ float schlick_weight(float cos_theta)
{
    auto const m{clamp(1.0f - cos_theta, 0.0f, 1.0f)};
    auto const m2{m * m};
    return m2 * m2 * m; // pow(m, 5)
}

inline __both__ float eta_to_f0(float eta)
{
    return sqr((1.0f - eta) / (1.0f + eta));
}

inline __both__ float relative_eta(vec3 const& wo, float ior,
                                  float& eta_i, float& eta_t)
{
    eta_i = cos_theta(wo) > 0.0f ? 1.0f : ior;
    eta_t = cos_theta(wo) > 0.0f ? ior : 1.0f;
    return eta_i / eta_t;
}

inline __both__ float roughness_to_alpha(float roughness)
{
    return max(alpha_min, clamp(sqr(roughness), 0.0f, 1.0f));
}

inline __both__ vec2 roughness_to_alpha(float roughness, float anisotropy)
{
    auto const aspect{sqrt(1.0f - 0.9f * anisotropy)};
    return {max(alpha_min, sqr(roughness) / aspect), max(alpha_min, sqr(roughness) * aspect)};
}

/// used for bsdf specular lobe
/// Disney 2015 - eq. (8)
inline __both__ float fresnel_equation(vec3 const& i, vec3 const& m, float eta_i, float eta_t)
{
    auto const c{abs(dot(i, m))};
    auto const denominator{sqr(eta_t / eta_i) - 1.0f + sqr(c)};
    if (denominator < 0.0f) return 1.0f; // total internal reflection
    auto const g{sqrt(denominator)};
    return 0.5f * sqr((g - c) / (g + c)) *
        (1.0f + sqr(c * (g + c) - 1.0f) / sqr(c * (g - c) + 1.0f));
}

#endif //PATH_TRACER_DISNEY_HELPER_CUH
