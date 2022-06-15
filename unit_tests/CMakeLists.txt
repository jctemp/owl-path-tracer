include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(path_tracer_test
        "path_tracer_test.cu"
        ${CMAKE_SOURCE_DIR}/path_tracer/src/sample_methods.hpp)

target_link_libraries(path_tracer_test gtest_main)
