#ifndef DEVICE_TYPES_HPP
#define DEVICE_TYPES_HPP
#pragma once

#include <owl/common/math/vec.h>
#include <owl/owl_device_buffer.h>

using Float = float;
using Float2 = owl::vec2f;
using Float3 = owl::vec3f;
using Float4 = owl::vec4f;

using Int = int32_t;
using Int2 = owl::vec2i;
using Int3 = owl::vec3i;

using Uint = uint32_t;
using Uint2 = owl::vec2ui;
using Uint3 = owl::vec3ui;

using Buffer = owl::device::Buffer;

#endif // !DEVICE_TYPES_HPP
