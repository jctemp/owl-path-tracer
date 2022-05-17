#pragma once

#include <owl/common/math/vec.h>
#include <owl/owl_device_buffer.h>

#include <fmt/color.h>

#define PI            3.14159265358979323f // pi
#define TWO_PI        6.28318530717958648f // 2pi
#define PI_OVER_TWO   1.57079632679489661f // pi / 2
#define PI_OVER_FOUR  0.78539816339744830f // pi / 4
#define INV_PI        0.31830988618379067f // 1 / pi
#define INV_TWO_PI    0.15915494309189533f // 1 / (2pi)
#define INV_FOUR_PI   0.07957747154594766f // 1 / (4pi)
#define EPSILON       1E-5f
#define T_MIN         1E-3f
#define T_MAX         1E10f
#define MIN_ROUGHNESS 0.01f
#define MIN_ALPHA     0.001f

namespace color
{
	static fmt::terminal_color constexpr log{ fmt::terminal_color::bright_cyan };
	static fmt::terminal_color constexpr warn{ fmt::terminal_color::yellow };
	static fmt::terminal_color constexpr error{ fmt::terminal_color::red };
	static fmt::terminal_color constexpr ok{ fmt::terminal_color::green };
	static fmt::terminal_color constexpr start{ fmt::terminal_color::bright_magenta };
	static fmt::terminal_color constexpr stop{ fmt::terminal_color::magenta };
}

using Float = float;
using Float2 = owl::vec2f;
using Float3 = owl::vec3f;
using Float4 = owl::vec4f;

using Int = int32_t;
using Int2 = owl::vec2i;
using Int3 = owl::vec3i;

using Uint = uint32_t;
using Uint64 = uint64_t;
using Uint2 = owl::vec2ui;
using Uint3 = owl::vec3ui;

using Buffer = owl::device::Buffer;

