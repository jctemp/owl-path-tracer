#ifndef PATH_TRACER_SAMPLE_METHODS_HPP
#define PATH_TRACER_SAMPLE_METHODS_HPP


#include "device.hpp"
#include "Math.hpp"

/*
* - http://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf.
* - https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
*/

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "GENERIC SAMPLING"

__device__ void sampleUniformDisk(vec2& rand)
{
    float phi{ two_pi * rand.y };
    float r{ owl::sqrt(rand.x) };

    rand.x = r * owl::cos(phi);
    rand.y = r * owl::sin(phi);
}


__device__ void sampleConcentricDisk(vec2& rand)
{
    // re-scale rand to be between [-1,1]
    float dx{ 2.0f * rand.x - 1 };
    float dy{ 2.0f * rand.y - 1 };

    // handle degenerated origin
    if (dx == 0 && dy == 0)
    {
        rand.x = 0;
        rand.y = 0;
        return;
    }

    // handle mapping unit squre to unit disk
    float phi, r;
    if (std::abs(dx) > std::abs(dy))
    {
        r = dx;
        phi = pi_over_four * (dy / dx);
    }
    else
    {
        r = dy;
        phi = pi_over_two - pi_over_four * (dx / dy);
    }

    rand.x = r * owl::cos(phi);
    rand.y = r * owl::sin(phi);
}


__device__ vec3 sampleUniformSphere(vec2& rand)
{
float z{ 1.0f - 2.0f * rand.x };
float r{ sqrtf(fmaxf(0.0f, 1.0f - z * z)) };
float phi{ two_pi * rand.y };
float x = r * owl::cos(phi);
float y = r * owl::sin(phi);

return vec3{ x, y, z };
}

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "COSINE HEMISPHERE SAMPLING"

__device__ void sampleCosineHemisphere(vec2 rand, vec3& L)
{
// 1. sample unit circle and save position into randu, randv
sampleConcentricDisk(rand);

// 2. calculate cosTheta => 1 = randu^2 + randv^2 => cos = 1 - (randu^2 + randv^2)
float cosTheta{ owl::sqrt(owl::max(0.0f, 1.0f - rand.x * rand.x - rand.y * rand.y)) };

L = vec3{ rand.x, rand.y, cosTheta };
}


__device__ void pdfCosineHemisphere(vec3 const& V, vec3 const& L, float& pdf)
{
    pdf = absCosTheta(L) * inv_pi;
}

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "UNIFORM HEMISPHERE SAMPLING"

__device__ void sampleUniformHemisphere(vec2 rand, vec3& L)
{
float z{ rand.x };
float r{ sqrtf(fmaxf(0.0f, 1.0f - z * z)) };
float phi = two_pi * rand.y;

float x = r * owl::cos(phi);
float y = r * owl::sin(phi);

L = vec3{ x, y, z };
}


__device__ void pdfUniformHemisphere(vec3 const& V, vec3 const& L, float& pdf)
{
    pdf = 0.5f * inv_pi;
}

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "CONE SAMPLING"

__device__ void sampleUniformCone(vec2 rand, float cosThetaMax, vec3& w)
{
float cosTheta{ (1.0f - rand.u) + rand.u * cosThetaMax };
float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
float phi{ rand.v * 2.0f * pi };
w = vec3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
}


__device__ void pdfUniformCone(float cosThetaMax, float& pdf)
{
    pdf = 1.0f / (2.0f * pi * (1.0f - cosThetaMax));
}

#pragma endregion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "TRIANGLE SAMPLING"

__device__ void sampleUniformTriangle(vec2 rand, vec2& p)
{
float su0{ sqrtf(rand.u) };
p = vec2{ 1 - su0, rand.v * su0 };
}

__device__ void sampleTriangle(InterfaceStruct const& i, Random& rand)
{
    vec2 b{};
    sampleUniformTriangle({ rand.random(), rand.random() }, b);

}

__device__ float triangleArea(vec3 const& A, vec3 const& B, vec3 const& C)
{
    float a{ owl::length((B - A)) };
    float b{ owl::length((C - A)) };
    float c{ owl::length((C - B)) };

    return 0.25f * sqrtf((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c));
}

#pragma endregion


#endif //PATH_TRACER_SAMPLE_METHODS_HPP
