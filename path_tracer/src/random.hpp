/*
 * Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SHARED_RANDOM_HPP
#define SHARED_RANDOM_HPP

#include "types.hpp"

struct random
{
    inline __both__ random() : state{0} { /* REQUIRED FOR DEVICE VARS */ }

    inline __both__ random(int32_t seed_u, int32_t seed_v) : state{0} { init((uint32_t) seed_u, (uint32_t) seed_v); }

    inline __both__ explicit random(uvec2 seed) : state{0} { init((uint32_t) seed.u, (uint32_t) seed.v); }

    inline __both__ random(uint32_t seed_u, uint32_t seed_v) : state{0} { init(seed_u, seed_v); }

    inline __both__ explicit random(ivec2 seed) : state{0} { init(seed.u, seed.v); }

    inline __both__ void init(uint32_t seed_u, uint32_t seed_v)
    {
        uint32_t s{0};
        for (int32_t n = 0; n < N; n++)
        {
            s += 0x9e3779b9;
            seed_u += ((seed_v << 4) + 0xa341316c) ^ (seed_v + s) ^ ((seed_v >> 5) + 0xc8013ea4);
            seed_v += ((seed_u << 4) + 0xad90777d) ^ (seed_u + s) ^ ((seed_u >> 5) + 0x7e95761e);
        }
        state = seed_u;
    }

    template<typename T = float>
    inline __both__ T rng();

    template<>
    inline __both__ float rng()
    {
        uint32_t constexpr A{16807};
        uint32_t constexpr C{1013904223};
        state = A * state + C;
        return ldexpf((float) state, -32);

    }

    template<>
    inline __both__ vec2 rng()
    {
        return {this->rng(), this->rng()};
    }

    template<typename T = float>
    inline __both__ float operator()()
    {
        return this->rng<T>();
    }

    int32_t N{4};
    uint32_t state;
};

#endif // ! SHARED_RANDOM_HPP
