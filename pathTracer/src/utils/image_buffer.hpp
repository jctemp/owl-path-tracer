#pragma once

#include <types.hpp>

#include <string>
#include <filesystem>

struct image_buffer
{
	enum class tag
	{
		allocated,
		referenced
	};

	image_buffer() : width{ 0 }, height{ 0 }, buffer{ nullptr }, ptr_tag{ tag::allocated }
	{
	}

	image_buffer(int32_t x, int32_t y, uint32_t* ptr, image_buffer::tag t)
		: width{ x }, height{ y }, buffer{ ptr }, ptr_tag{ t }
	{
	}

	~image_buffer()
	{
		if (ptr_tag == tag::allocated && buffer != nullptr)
			delete[] buffer;
	}

	int32_t width;
	int32_t height;
	uint32_t* buffer;
	tag ptr_tag;
};

extern void write_image(image_buffer const& i, std::string const& name = "image", std::string const& path = ".");

extern image_buffer load_image(std::string const& name = "image", std::string const& path = ".");