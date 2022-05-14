SET(BA_SOURCES
	"Main.cpp"
	"src/host/ObjLoader.cpp"
	"src/host/StbUtils.cpp"
	"src/host/Renderer.cpp"
)

SET(BA_HEADERS
	
	"src/host/ObjLoader.hpp"
	"src/host/StbUtils.hpp"
	"src/host/Renderer.hpp"
	
)

SET(BA_DEVICE_HEADERS
	"src/device/core/closestHit.hpp"
	"src/device/core/miss.hpp"

    "src/device/utils/types.hpp"
    "src/device/utils/random.hpp"
	 "src/device/utils/macros.hpp"

	"src/device/Shared.hpp"

	
	"src/device/Math.hpp" 

	"src/device/Globals.hpp"
	"src/device/Sampling.hpp"

	"src/device/materials/Lambert.hpp"
	"src/device/materials/Disney.hpp"
    "src/device/materials/BrdfUtils.hpp"
)

SET(BA_INCLUDE "include" "src")

EMBED_PTX(
	OUTPUT_TARGET
		device-ptx
	PTX_LINK_LIBRARIES
		owl::owl
		${CUDA_RUNTIME_LIBRARY}
	SOURCES
		"src/device/device.cu"
)



add_executable(bachelorThesis ${BA_SOURCES} ${BA_HEADERS} ${BA_DEVICE_HEADERS})
target_include_directories(bachelorThesis PUBLIC ${BA_INCLUDE})
target_link_libraries(bachelorThesis
	owl::owl
	fmt::fmt
	glm::glm
	tinyobjloader
	device-ptx
)

if(MSVC)
	# not receiving errors due to msvc safe functions
	add_compile_definitions(_CRT_SECURE_NO_WARNINGS=1)
endif()