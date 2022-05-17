# BASIC PROJECT SETTINGS
cmake_minimum_required (VERSION 3.8)

project ("bachelorThesis" VERSION 0.3.0 LANGUAGES CXX C CUDA)

# SET STANDARDS

if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 20)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
	set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(NOT DEFINED CMAKE_C_STANDARD)
	set(CMAKE_C_STANDARD 17)
	set(CMAKE_C_STANDARD_REQUIRED ON)
	set(CMAKE_C_EXTENSIONS OFF)
endif()

if(MSVC)
	# not receiving errors due to msvc safe functions
	add_compile_definitions(_CRT_SECURE_NO_WARNINGS=1)
endif()

# CONFIGURE EXTERNAL LIBRARIES
set(log_dir "${CMAKE_CURRENT_SOURCE_DIR}/simpleLogger")
include_directories(${log_dir})
add_subdirectory(${log_dir} EXCLUDE_FROM_ALL)

set(OptiX_ROOT_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0")

set(OWL_BUILD_SAMPLES OFF CACHE BOOL "Build the Samples?")
set(owl_dir "external/owl")
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

set(fmt_dir "external/fmt")
add_subdirectory(${fmt_dir} EXCLUDE_FROM_ALL)

set(obj_dir "external/tinyobjloader")
add_subdirectory(${obj_dir} EXCLUDE_FROM_ALL)

set(stb_dir "external/stb")
include_directories(${stb_dir} EXCLUDE_FROM_ALL)

set(glm_dir "external/glm")
add_subdirectory(${glm_dir})

set(json_dir "external/json")
add_subdirectory(${json_dir} EXCLUDE_FROM_ALL)

add_subdirectory ("${CMAKE_CURRENT_SOURCE_DIR}/pathTracer")
