
#ifndef PATH_TRACER_MATH_HPP
#define PATH_TRACER_MATH_HPP

#include <owl/common.h>
#include <types.hpp>

namespace math
{
    inline __both__ float mix(float a, float b, float t);

    inline __both__ vec3 mix(vec3 a, vec3 b, float t);

    inline __both__ float clamp(float v, float min, float max);

    inline __both__ vec3 clamp(vec3 v, vec3 min, vec3 max);

    inline __both__ float saturate(float v);

    inline __both__ vec3 saturate(vec3 v);

    inline __both__ float sqrt(float v);

    inline __both__ vec3 sqrt(vec3 v);

    inline __both__ float sqr(float v);

    inline __both__ float min(float a, float b);

    inline __both__ float max(float a, float b);

    inline __both__ float sin(float v);

    inline __both__ float asin(float v);

    inline __both__ float cos(float v);

    inline __both__ float acos(float v);

    inline __both__ float tan(float v);

    inline __both__ float atan(float v);

    namespace geometric
    {
        __both__ void onb(vec3 const& N, vec3& T, vec3& B);

        // move vector V to local space where N is (0,0,1)
        inline __both__ void toLocal(vec3 const& T, vec3 const& B, vec3 const& N, vec3& V);

        // move V from local to the global space
        inline __both__ void toWorld(vec3 const& T, vec3 const& B, vec3 const& N, vec3& V);
    }

    namespace shading
    {

        inline __both__ float cos_2_theta(vec3 const& w);

        inline __both__ float abs_cos_theta(vec3 const& w);

        inline __both__ float sin_2_theta(vec3 const& w);

        inline __both__ float sin_theta(vec3 const& w);

        inline __both__ float tan_theta(vec3 const& w);

        inline __both__ float tan_2_theta(vec3 const& w);

        inline __both__ float cos_phi(vec3 const& w);

        inline __both__ float sin_phi(vec3 const& w);

        inline __both__ float cos_2_phi(vec3 const& w);

        inline __both__ float sin_2_phi(vec3 const& w) ;

        inline __both__ float cos_d_phi(vec3 const& wa, vec3 const& wb);

        inline __both__ vec3 reflect(vec3 const& w, vec3 const& N);

        inline __both__ bool same_hemisphere(vec3 const& a, vec3 const& b);

    }

    namespace constants
    {
        float constexpr t_min{1E-3f};
        float constexpr t_max{1E10f};
        float constexpr alpha_min{1E-3f};

        float constexpr pi{3.14159265358979323f};
        float constexpr two_pi{2.0f * pi};
        float constexpr pi_div_two{pi / 2.0f};
        float constexpr pi_div_four{pi / 4.0f};
        float constexpr inverse_pi{1.0f / pi};
        float constexpr inverse_two_pi{1.0f / (2.0f * pi)};
        float constexpr inverse_four_pi{1.0f / (4.0f * pi)};
    }

}

#endif //PATH_TRACER_MATH_HPP
