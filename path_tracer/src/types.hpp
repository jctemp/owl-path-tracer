
#ifndef PATH_TRACER_TYPES_HPP
#define PATH_TRACER_TYPES_HPP

#include <owl/common/math/vec.h>
#include <owl/owl_device_buffer.h>
#include <fmt/color.h>

__constant__ auto constexpr pi{3.14159265358979323f};
__constant__ auto constexpr two_pi{6.28318530717958648f};
__constant__ auto constexpr pi_over_two{1.57079632679489661f};
__constant__ auto constexpr pi_over_four{0.78539816339744830f};
__constant__ auto constexpr inv_pi{0.31830988618379067f};
__constant__ auto constexpr inv_two_pi{0.15915494309189533f};
__constant__ auto constexpr inv_four_pi{0.07957747154594766f};
__constant__ auto constexpr t_min{1E-3f};
__constant__ auto constexpr t_max{1E10f};
__constant__ auto constexpr alpha_min{0.001f};

namespace color
{
    static fmt::terminal_color constexpr log{fmt::terminal_color::bright_cyan};
    static fmt::terminal_color constexpr warn{fmt::terminal_color::yellow};
    static fmt::terminal_color constexpr error{fmt::terminal_color::red};
    static fmt::terminal_color constexpr ok{fmt::terminal_color::green};
    static fmt::terminal_color constexpr start{fmt::terminal_color::bright_magenta};
    static fmt::terminal_color constexpr stop{fmt::terminal_color::magenta};
}

using vec2 = owl::vec2f;
using vec3 = owl::vec3f;
using ivec2 = owl::vec2i;
using ivec3 = owl::vec3i;
using uvec2 = owl::vec2ui;
using uvec3 = owl::vec3ui;

using Buffer = owl::device::Buffer;

#endif //PATH_TRACER_TYPES_HPP
