#ifndef CAMERA_HPP
#define CAMERA_HPP
#pragma once

#include <types.hpp>

struct camera
{
    owl::vec3f const look_from;
    owl::vec3f const look_at;
    owl::vec3f const look_up;
    float const vertical_fov;
};

struct camera_data
{
    owl::vec3f origin;
    owl::vec3f llc;
    owl::vec3f horizontal;
    owl::vec3f vertical;
};

extern camera_data to_camera_data(camera const& c, owl::vec2i const& buffer_size);

#endif // !CAMERA_HPP
