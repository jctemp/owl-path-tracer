#include <pathTracer/StbUtils.hpp>
#include <simpleLogger.hpp>

#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

namespace ba
{
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

	void loadImage(ImageRgb& i, std::string const& name, std::string const& path)
	{
		SL_LOG(fmt::format("Try to load image {}/{} ", path, name));

		std::string fullIdentifier{ fmt::format("{}/{}", path, name) };

		int32_t comp;
		auto buffer{ stbi_load(fullIdentifier.c_str(), &i.width, &i.height, &comp, STBI_rgb) };

		i.texels.resize(i.width * i.height * 4);
		for (int32_t y{ i.height - 1 }; y >= 0; --y) {
			for (int32_t x{ 0 }; x < i.width; ++x) {
				int32_t index{ (y * i.width + x) * 4 };
				i.texels[index] = *buffer++;
				i.texels[index + 1] = *buffer++;
				i.texels[index + 2] = *buffer++;
				i.texels[index + 3] = (comp == 3) ? 1U : *buffer++;
			}
		}
		SL_OK("Successfully loaded texture...");
	}
}