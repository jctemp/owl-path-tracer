#ifndef STB_UTILS_HPP
#define STB_UTILS_HPP

#include <cinttypes>
#include <string>
#include <vector>

struct Image
{
	int32_t width;
	int32_t height;
	uint32_t const* buffer;
};

struct ImageRgb
{
	~ImageRgb()
	{
		if (pixel) delete[] pixel;
	}
	int32_t width;
	int32_t height;
	uint32_t* pixel{ nullptr };
};

void writeImage(Image const& i, std::string const& name, std::string const& destinationPath = ".");
	
void loadImage(ImageRgb& i, std::string const& name, std::string const& path = ".");

#endif // !BA_STB_UTILS_HPP
