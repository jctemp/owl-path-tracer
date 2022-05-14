#include <SimpleLogger.hpp>
#include "StbUtils.hpp"

#include <filesystem>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>


void writeImage(Image const& i, std::string const& name, std::string const& path)
{
	if (!i.buffer)
	{
		SL_ERROR("Buffer is nullptr...");
		throw std::runtime_error{ "buffer is null" };
	}

	std::string fullIdentifier{ fmt::format("{}/{}", path, name) };
	SL_LOG(fmt::format("Writing image {}", fullIdentifier));
	stbi_write_png(fullIdentifier.c_str(), i.width, i.height, 4, i.buffer, i.width * sizeof(uint32_t));
	SL_OK("Image written...");
}

void loadImage(ImageRgb& tex, std::string const& name, std::string const& path)
{
	SL_LOG(fmt::format("Try to load image {}/{} ", path, name));

	std::string fullIdentifier{ fmt::format("{}/{}", path, name) };

	if (!std::filesystem::exists(fullIdentifier))
	{
		SL_ERROR(fmt::format("{} does not exists at the location", fullIdentifier));
		exit(1);
	}

	int32_t comp;
	auto buffer{ stbi_load(fullIdentifier.c_str(), &tex.width,
		&tex.height, &comp, STBI_rgb_alpha) };

	tex.pixel = (uint32_t*)buffer;

	for (int32_t y{ 0 }; y < tex.height / 2; y++) 
	{
		uint32_t* line_y{ tex.pixel + y * tex.width };
		uint32_t* mirrored_y{ tex.pixel + (tex.height - 1 - y) * tex.width };
		for (int x = 0; x < tex.width; x++) 
		{
			std::swap(line_y[x], mirrored_y[x]);
		}
	}

	SL_OK("Successfully loaded texture...");
}
