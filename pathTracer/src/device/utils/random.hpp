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

#ifndef DEVICE_RANDOM_HPP
#define DEVICE_RANDOM_HPP
#pragma once

#include "types.hpp"
#include "macros.hpp"

struct Random
{
	PT_BOTH_INLINE Random() { /* REQUIRED FOR DEVICE VARS */ }

	PT_BOTH_INLINE Random(Int seedu, Int seedv) { init((Uint)seedu, (Uint)seedv); }
	PT_BOTH_INLINE Random(Int2 seed) { init((Uint)seed.u, (Uint)seed.v); }

	PT_BOTH_INLINE Random(Uint seedu, Uint seedv) { init(seedu, seedv); }
	PT_BOTH_INLINE Random(Uint2 seed) { init(seed.u, seed.v); }

	PT_BOTH_INLINE void init(Uint seedu, Uint seedv)
	{
		Uint s{ 0 };
		for (Uint n = 0; n < N; n++) {
			s += 0x9e3779b9;
			seedu += ((seedv << 4) + 0xa341316c) ^ (seedv + s) ^ ((seedv >> 5) + 0xc8013ea4);
			seedv += ((seedu << 4) + 0xad90777d) ^ (seedu + s) ^ ((seedu >> 5) + 0x7e95761e);
		}
		state = seedu;
	}

	template<typename T = Float>
	PT_BOTH_INLINE T random();

	template<>
	PT_BOTH_INLINE Float random()
	{
		Uint constexpr A{ 16807 };
		Uint constexpr C{ 1013904223 };
		state = A * state + C;
		return ldexpf((Float)state, -32);

	}

	template<>
	PT_BOTH_INLINE Float2 random()
	{
		return { this->random(), this->random() };
	}

	template<typename T = Float>
	PT_BOTH_INLINE Float operator()()
	{
		return this->random<T>();
	}

	Int N{4};
	Uint state;
};

#endif // ! DEVICE_RANDOM_HPP
