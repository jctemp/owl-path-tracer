#ifndef CAMERA_HPP
#define CAMERA_HPP
#pragma once

#include <types.hpp>

struct camera
{
    glm::vec3 const look_from;
    glm::vec3 const look_at;
    glm::vec3 const look_up;
    float const vertical_fov;
};

struct camera_data
{
    glm::vec3 origin;
    glm::vec3 llc;
    glm::vec3 horizontal;
    glm::vec3 vertical;
};

extern camera_data to_camera_data(camera const& c, glm::ivec2 const& buffer_size);

#endif // !CAMERA_HPP
