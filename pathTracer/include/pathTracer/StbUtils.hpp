#ifndef BA_STB_UTILS_HPP
#define BA_STB_UTILS_HPP

#include <cinttypes>
#include <string>
#include <vector>

namespace ba
{
	struct Image
	{
		int32_t width;
		int32_t height;
		uint32_t const* buffer;
	};

	struct ImageRgb
	{
		int32_t width;
		int32_t height;
		std::vector<uint8_t> texels;
	};

	void writeImage(Image const& i, std::string const& name, std::string const& destinationPath = ".");
	
	void loadImage(ImageRgb& i, std::string const& name, std::string const& path = ".");
}

#endif // !BA_STB_UTILS_HPP
