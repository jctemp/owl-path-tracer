# BASIC PROJECT SETTINGS
cmake_minimum_required (VERSION 3.8)

project ("bachelorThesis" VERSION 0.1.0 LANGUAGES CXX C CUDA)

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

# CONFIGURE EXTERNAL LIBRARIES

set(log_dir "${CMAKE_CURRENT_SOURCE_DIR}/simpleLogger")
include_directories(${log_dir})
add_subdirectory(${log_dir} EXCLUDE_FROM_ALL)

set(OptiX_ROOT_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0")

set(owl_dir "${CMAKE_CURRENT_SOURCE_DIR}/external/owl")
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

set(fmt_dir "${CMAKE_CURRENT_SOURCE_DIR}/external/fmt")
add_subdirectory(${fmt_dir} EXCLUDE_FROM_ALL)

set(obj_dir "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobjloader")
add_subdirectory(${obj_dir} EXCLUDE_FROM_ALL)

set(stb_dir "${CMAKE_CURRENT_SOURCE_DIR}/external/stb")
include_directories(${stb_dir} EXCLUDE_FROM_ALL)

# Include sub-projects.
add_subdirectory ("${CMAKE_CURRENT_SOURCE_DIR}/experiments")
add_subdirectory ("${CMAKE_CURRENT_SOURCE_DIR}/rayTracer")
