

SET(HOST_SOURCES
	"Main.cpp"
	
	"src/utils/image_buffer.hpp"
	"src/utils/image_buffer.cpp"
	"src/utils/mesh_loader.hpp"
	"src/utils/mesh_loader.cpp"

	"src/host/Renderer.cpp"	
)

SET(SHARED_SOURCES
	"src/shared/shared.hpp"
    "src/shared/types.hpp" 
    "src/shared/random.hpp"
	"src/shared/macros.hpp"
)

SET(DEVICE_SOURCES
	"src/device/device.hpp"

	"src/device/core/core.hpp"
	"src/device/core/closestHit.hpp"
	"src/device/core/miss.hpp"
	"src/device/core/pathTracer.hpp"

	"src/device/Math.hpp" 
	"src/device/Sampling.hpp"
	"src/device/materials/Lambert.hpp"
	"src/device/materials/Disney.hpp"
    "src/device/materials/BrdfUtils.hpp"
 )

SET(INCLUDE_DIRS "include" "src")
EMBED_PTX(
	OUTPUT_TARGET device-ptx
	PTX_LINK_LIBRARIES owl::owl glm::glm ${CUDA_RUNTIME_LIBRARY}
	SOURCES "src/device/device.cu"
)



add_executable(bachelorThesis ${SOURCES_UTLS} ${HOST_SOURCES} ${DEVICE_SOURCES} ${SHARED_SOURCES})
target_include_directories(bachelorThesis PUBLIC ${INCLUDE_DIRS})
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