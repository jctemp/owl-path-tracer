
#include "math.hpp"


namespace math
{
    inline __both__ float mix(float a, float b, float t);

    inline __both__ vec3 mix(vec3 a, vec3 b, float t);

    inline __both__ float clamp(float v, float min, float max) { return owl::clamp(v, min, max); }

    inline __both__ vec3 clamp(vec3 v, vec3 min, vec3 max) { return owl::clamp(v, min, max); }

    inline __both__ float saturate(float v) { return clamp(v, 0.0, 1.0); }

    inline __both__ vec3 saturate(vec3 v) { return clamp(v, vec3{0.0}, vec3{1.0}); }

    inline __both__ float sqrt(float v) { return owl::sqrt(v); }

    inline __both__ vec3 sqrt(vec3 v) { return owl::sqrt(v); }

    inline __both__ float sqr(float v) { return v * v; }

    inline __both__ float min(float a, float b) { return owl::min(a, b); }

    inline __both__ float max(float a, float b) { return owl::max(a, b); }

    inline __both__ float sin(float v) { return sinf(v); }

    inline __both__ float asin(float v) { return asinf(v); }

    inline __both__ float cos(float v) { return cosf(v); }

    inline __both__ float acos(float v) { return acosf(v); }

    inline __both__ float tan(float v) { return tanf(v); }

    inline __both__ float atan(float v) { return atanf(v); }

    namespace geometric
    {
        __both__ void onb(vec3 const& N, vec3& T, vec3& B)
        {
            if (N.x != N.y || N.x != N.z)
                T = vec3(N.z - N.y, N.x - N.z, N.y - N.x);    // ( 1, 1, 1) x N
            else
                T = vec3(N.z - N.y, N.x + N.z, -N.y - N.x);   // (-1, 1, 1) x N

            T = normalize(T);
            B = cross(N, T);
        }

        // move vector V to local space where N is (0,0,1)
        inline __both__ void toLocal(vec3 const& T, vec3 const& B, vec3 const& N, vec3& V)
        {
            V = normalize(vec3{dot(V, T), dot(V, B), dot(V, N)});
        }

        // move V from local to the global space
        inline __both__ void toWorld(vec3 const& T, vec3 const& B, vec3 const& N, vec3& V)
        {
            V = normalize(vec3{V.x * T + V.y * B + V.z * N});
        }
    }

    namespace shading
    {

        inline __both__ float cos_theta(vec3 const& w) { return w.z; }

        inline __both__ float cos_2_theta(vec3 const& w) { return w.z * w.z; }

        inline __both__ float abs_cos_theta(vec3 const& w) { return abs(w.z); }

        inline __both__ float sin_2_theta(vec3 const& w) { return max(0.0f, 1.0f - cos_2_theta(w)); }

        inline __both__ float sin_theta(vec3 const& w) { return sqrt(sin_2_theta(w)); }

        inline __both__ float tan_theta(vec3 const& w) { return sin_theta(w) / cos_theta(w); }

        inline __both__ float tan_2_theta(vec3 const& w) { return sin_2_theta(w) / cos_2_theta(w); }

        inline __both__ float cos_phi(vec3 const& w)
        {
            float theta{sin_theta(w)};
            return (theta == 0) ? 1.0f : clamp(w.x / theta, -1.0f, 1.0f);
        }

        inline __both__ float sin_phi(vec3 const& w)
        {
            float theta{sin_theta(w)};
            return (theta == 0) ? 1.0f : clamp(w.y / theta, -1.0f, 1.0f);
        }

        inline __both__ float cos_2_phi(vec3 const& w) { return cos_phi(w) * cos_phi(w); }

        inline __both__ float sin_2_phi(vec3 const& w) { return sin_phi(w) * sin_phi(w); }

        inline __both__ float cos_d_phi(vec3 const& wa, vec3 const& wb)
        {
            return clamp((wa.x * wb.x + wa.y * wb.y) /
                         sqrt((wa.x * wa.x + wa.y * wa.y) *
                              (wb.x * wb.x + wb.y * wb.y)), -1.0f, 1.0f);
        }

        inline __both__ vec3 reflect(vec3 const& w, vec3 const& N) { return (2.0f * dot(w, N)) * N - w; }

        inline __both__ bool same_hemisphere(vec3 const& a, vec3 const& b) { return a.z * b.z > 0.0f; }

    }