set(BA_SOURCES
	"src/Main.cpp"
	"src/ObjLoader.cpp"
	"src/StbUtils.cpp"
)

SET(BA_HEADERS
	"include/pathTracer/ObjLoader.hpp"
	"include/pathTracer/OWLRenderer.hpp"
	"include/pathTracer/Renderer.hpp"
	"include/pathTracer/StbUtils.hpp"
)

SET(BA_DEVICE_HEADERS
	"src/device/PathTracer.hpp"
	"src/device/Random.hpp"
	"src/device/Types.hpp"
)

SET(BA_INCLUDE "./include" "./src")

EMBED_PTX(
	OUTPUT_TARGET
		pathTracer-ptx
	PTX_LINK_LIBRARIES
		owl::owl
		${CUDA_RUNTIME_LIBRARY}
	SOURCES
		"./src/device/PathTracer.cu"
)



add_executable(bachelorThesis ${BA_SOURCES} ${BA_HEADERS} ${BA_DEVICE_HEADERS})
target_include_directories(bachelorThesis PUBLIC ${BA_INCLUDE})
target_link_libraries(bachelorThesis
	owl::owl
	fmt::fmt
	glm::glm
	tinyobjloader
	pathTracer-ptx
)

if(MSVC)
	# not receiving errors due to msvc safe functions
	add_compile_definitions(_CRT_SECURE_NO_WARNINGS=1)
endif()