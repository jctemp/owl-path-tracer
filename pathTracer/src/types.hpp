#pragma once

#include <owl/common/math/vec.h>
#include <owl/owl_device_buffer.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <fmt/color.h>

static struct color
{
	static fmt::terminal_color constexpr log{ fmt::terminal_color::bright_cyan };
	static fmt::terminal_color constexpr warn{ fmt::terminal_color::yellow };
	static fmt::terminal_color constexpr error{ fmt::terminal_color::red };
	static fmt::terminal_color constexpr ok{ fmt::terminal_color::green };
	static fmt::terminal_color constexpr start{ fmt::terminal_color::bright_magenta };
	static fmt::terminal_color constexpr stop{ fmt::terminal_color::magenta };
};


inline __both__ owl::vec2f make_owl_type(glm::vec2 const& v) { return  owl::vec2f{ v.x, v.y }; }
inline __both__ owl::vec3f make_owl_type(glm::vec3 const& v) { return  owl::vec3f{ v.x, v.y, v.z }; }
inline __both__ owl::vec2i make_owl_type(glm::ivec2 const& v) { return owl::vec2i{ v.x, v.y }; }
inline __both__ owl::vec3i make_owl_type(glm::ivec3 const& v) { return owl::vec3i{ v.x, v.y, v.z }; }
inline __both__ owl::vec2ui make_owl_type(glm::uvec2 const& v) { return owl::vec2ui{ v.x, v.y }; }
inline __both__ owl::vec3ui make_owl_type(glm::uvec3 const& v) { return owl::vec3ui{ v.x, v.y, v.z }; }

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