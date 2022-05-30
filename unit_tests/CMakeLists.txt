include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(sample_methods_test
        "sample_methods_test.cpp"
        ${CMAKE_SOURCE_DIR}/path_tracer/src/sample_methods.hpp)

target_link_libraries(sample_methods_test gtest_main)
