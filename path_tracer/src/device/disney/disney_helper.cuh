#ifndef PATH_TRACER_DISNEY_HELPER_CUH
#define PATH_TRACER_DISNEY_HELPER_CUH

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

inline __both__ float roughness_to_alpha(float roughness)
{
    return max(alpha_min, clamp(sqr(roughness), 0.0f, 1.0f));
}

inline __both__ vec2 roughness_to_alpha(float roughness, float anisotropy)
{
    auto const aspect{sqrt(1.0f - 0.9f * anisotropy)};
    return {max(alpha_min, sqr(roughness) / aspect), max(alpha_min, sqr(roughness) * aspect)};
}

/// used for brdf specular lobe
/// Disney 2015 - eq. (7)
inline __both__ float fresnel_schlick(float cos_theta_i, float eta)
{
    auto const f0{powf((eta - 1.0f) / (eta + 1.0f), 2.0f)};
    auto const cos_theta2{1 - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta)};
    if (cos_theta2 <= 1.0f) return 1.0f;
    return f0 + (1.0f - f0) * schlick_weight(cos_theta2);
}

/// used for bsdf specular lobe
/// Disney 2015 - eq. (8)
inline __both__ float fresnel_equation(float cos_theta_i, float eta)
{
    auto const cos_theta2{1 - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta)};
    if (cos_theta2 <= 1.0f) return 1.0f;
    auto const cos_theta_t{owl::sqrt(cos_theta2)};
    auto s {(cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t)};
    auto t {(cos_theta_t - eta * cos_theta_i) / (cos_theta_t + eta * cos_theta_i)};
    s = s * s;
    t = t * t;
    return 0.5f * (s + t);
}

#endif //PATH_TRACER_DISNEY_HELPER_CUH
