
#ifndef PATH_TRACER_MACROS_HPP
#define PATH_TRACER_MACROS_HPP

#define get_data(dest, buffer, address, type) \
	if (buffer.data == nullptr) { \
		::printf("Device Error (%d, %s): buffer was nullptr.\n", __LINE__, __FILE__); asm("trap;"); } \
	if (address >= buffer.count) { \
		::printf("Device Error (%d, %s): out of bounds access (address: %d, size %d).\n", \
			__LINE__, __FILE__, address, uint32_t(buffer.count)); asm("trap;"); } \
	dest = ((type*)buffer.data)[address];

#define assert_condition(check, msg) \
	if (check) {::printf("Device Error (%d, %s): %s\n", __LINE__, __FILE__, msg); asm("trap;"); }

#endif //PATH_TRACER_MACROS_HPP

