#ifndef SIMPLE_LOGGER_HPP
#define SIMPLE_LOGGER_HPP

#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include <fmt/color.h>

#define S_MESSAGE fmt::terminal_color::cyan
#define S_SUCCESS fmt::terminal_color::green
#define S_WARNING fmt::terminal_color::yellow
#define S_ERROR fmt::terminal_color::red

#define LOG(msg) \
	fmt::print(fmt::fg(S_MESSAGE), "LOCATION[{}] IN LINE[{}]: {}", \
		__FILE__, __LINE__, msg); \
	fmt::print(fmt::fg(fmt::color::white), "\n")

#define OK(msg) \
	fmt::print(fmt::fg(S_SUCCESS), "LOCATION[{}] IN LINE[{}]: {}", \
		__FILE__, __LINE__, msg); \
	fmt::print(fmt::fg(fmt::color::white), "\n")

#define WARN(msg) \
	fmt::print(fmt::fg(S_WARNING), "LOCATION[{}] IN LINE[{}]: {}", \
		__FILE__, __LINE__, msg); \
	fmt::print(fmt::fg(fmt::color::white), "\n")

#define ERROR(msg) \
	fmt::print(fmt::fg(S_ERROR), "LOCATION[{}] IN LINE[{}]: {}", \
		__FILE__, __LINE__, msg); \
	fmt::print(fmt::fg(fmt::color::white), "\n")

#endif // SIMPLE_LOGGER_HPP