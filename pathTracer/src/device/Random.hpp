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

#ifndef BA_RANDOM_HPP
#define BA_RANDOM_HPP

#include <cinttypes>
#include <owl/common/math/vec.h>


/*! simple 24-bit linear congruence generator */
template<unsigned int N = 4>
struct LCG {

    inline __both__ LCG()
    { /* intentionally empty so we can use it in device vars that
            don't allow dynamic initialization (ie, PRD) */
    }
    inline __both__ LCG(unsigned int val0, unsigned int val1)
    {
        init(val0, val1);
    }

    inline __both__ LCG(const owl::vec2i& seed)
    {
        init((unsigned)seed.x, (unsigned)seed.y);
    }
    inline __both__ LCG(const owl::vec2ui& seed)
    {
        init(seed.x, seed.y);
    }

    inline __both__ void init(unsigned int val0, unsigned int val1)
    {
        unsigned int v0 = val0;
        unsigned int v1 = val1;
        unsigned int s0 = 0;

        for (unsigned int n = 0; n < N; n++) {
            s0 += 0x9e3779b9;
            v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
            v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
        }
        state = v0;
    }

    /// <summary>
    /// Generate random unsigned int in [0, 2^24) and rescales it between (-1,1)
    /// </summary>
    /// <returns>floating point value</returns>
    inline __both__ float random()
    {
        const uint32_t LCG_A = 1664525u;
        const uint32_t LCG_C = 1013904223u;
        state = (LCG_A * state + LCG_C);
        return ldexpf(float(state), -32);
        // return (state & 0x00FFFFFF) / (float) 0x01000000;
    }

    inline __both__ float operator()()
    {
        return this->random();
    }

    uint32_t state;
};

extern LCG<4> random;

#endif // !BA_RANDOM_HPP
