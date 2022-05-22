#include "image_buffer.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stb_image.h>
#include <stb_image_write.h>
#include <filesystem>

image_buffer::image_buffer() : width{0}, height{0}, buffer{nullptr}, ptr_tag{tag::allocated}
{
}

image_buffer::image_buffer(int32_t x, int32_t y, uint32_t const* ptr, image_buffer::tag t)
        : width{x}, height{y}, buffer{ptr}, ptr_tag{t}
{
}

image_buffer::~image_buffer()
{
    if (ptr_tag == tag::allocated && buffer != nullptr)
        delete[] buffer;
}

void write_image(image_buffer const& i, std::string const& name, std::string const& path)
{
    if (!i.buffer)
        throw std::runtime_error{"image buffer is null"};

    std::string const file_path{path + "/" + name};
    stbi_write_png(file_path.c_str(), i.width, i.height, 4,
            i.buffer, i.width * static_cast<int32_t>(sizeof(uint32_t)));
    printf("Image written to %s\n", std::filesystem::absolute(file_path).string().c_str());
}

image_buffer load_image(std::string const& name, std::string const& path)
{
    std::string const file_path{path + "/" + name};

    if (!std::filesystem::exists(file_path))
    {
        fmt::print(fg(color::warn), "Image file {} does not exist. Continue with empty.\n", file_path);
        return image_buffer{};
    }

    int32_t width, height, comp;
    auto buffer{reinterpret_cast<uint32_t*>(stbi_load(file_path.c_str(), &width,
                    &height, &comp, STBI_rgb_alpha))};

    for (int32_t y{0}; y < height / 2; y++)
    {
        uint32_t* line_y{buffer + y * width};
        uint32_t* mirrored_y{buffer + (height - 1 - y) * width};
        for (int x = 0; x < width; x++) std::swap(line_y[x], mirrored_y[x]);
    }

    image_buffer image{width, height, buffer, image_buffer::tag::manual};
    return image;
}