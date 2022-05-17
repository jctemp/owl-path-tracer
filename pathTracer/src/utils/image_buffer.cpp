#include "image_buffer.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

void write_image(image_buffer const& i, std::string const& name, std::string const& path)
{
    if (!i.buffer)
        throw std::runtime_error{ "image buffer is null" };

    std::string const file_path{ path + "/" + name };
    stbi_write_png(file_path.c_str(), i.width, i.height, 4, 
        i.buffer, i.width * sizeof(uint32_t));
    printf("Image written to %s\n", std::filesystem::absolute(file_path).string().c_str());
}

image_buffer load_image(std::string const& name, std::string const& path)
{
    std::string const file_path{ path + "/" + name };

    if (!std::filesystem::exists(file_path))
        throw std::runtime_error{ file_path + " does not exists at the location.\n" };

    image_buffer image{};

    int32_t comp;
    image.buffer = reinterpret_cast<uint32_t*>(stbi_load(file_path.c_str(), &image.width,
        &image.height, &comp, STBI_rgb_alpha));

    for (int32_t y{ 0 }; y < image.height / 2; y++)
    {
        uint32_t* line_y{ image.buffer + y * image.width };
        uint32_t* mirrored_y{ image.buffer + (image.height - 1 - y) * image.width };
        for (int x = 0; x < image.width; x++) std::swap(line_y[x], mirrored_y[x]);
    }

    return image;
}