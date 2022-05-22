
#ifndef PATH_TRACER_IMAGE_BUFFER_HPP
#define PATH_TRACER_IMAGE_BUFFER_HPP

#include <string>
#include "types.hpp"

struct image_buffer
{
    enum class tag
    {
        manual,
        allocated,
        referenced
    };

    image_buffer();

    image_buffer(int32_t x, int32_t y, uint32_t const* ptr, image_buffer::tag t);

    ~image_buffer();

    int32_t width;
    int32_t height;
    uint32_t const* buffer;
    tag ptr_tag;
};

extern void write_image(image_buffer const& i, std::string const& name = "image", std::string const& path = ".");

extern image_buffer load_image(std::string const& name = "image", std::string const& path = ".");

#endif //PATH_TRACER_IMAGE_BUFFER_HPP