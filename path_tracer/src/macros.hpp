
#ifndef PATH_TRACER_MACROS_HPP
#define PATH_TRACER_MACROS_HPP

#define get_data(dest, buffer, address, type) \
    if (buffer.data == nullptr) { \
        ::printf("Device Error (%d, %s): buffer was nullptr.\n", __LINE__, __FILE__); asm("trap;"); } \
    if (address >= buffer.count) { \
        ::printf("Device Error (%d, %s): out of bounds access (address: %d, size %d).\n", \
            __LINE__, __FILE__, address, uint32_t(buffer.count)); asm("trap;"); } \
    dest = ((type*)buffer.data)[address]

#define assert_condition(check, msg) \
    if (check) {::printf("Device Error (%d, %s): %s\n", __LINE__, __FILE__, msg); asm("trap;"); }

#define all_zero(vec) \
    (vec.x == 0.0f && vec.y == 0.0f && vec.z == 0.0f)

#define has_nan(vec) \
    (isnan(vec.x) || isnan(vec.y) || isnan(vec.z))

#define has_inf(vec) \
    (isinf(vec.x) || isinf(vec.y) || isinf(vec.z))

#endif //PATH_TRACER_MACROS_HPP

