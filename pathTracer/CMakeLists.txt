IF(MSVC)
	# not receiving errors due to msvc safe functions
	add_compile_definitions(_CRT_SECURE_NO_WARNINGS=1)
ENDIF()

SET(HEADER_FILES
	"src/device/camera.hpp"

	"src/utils/image_buffer.hpp"
	"src/utils/mesh_loader.hpp"

	"src/owl.hpp"

	# ------------------------------

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

	"src/shared/shared.hpp"
    "src/shared/types.hpp" 
    "src/shared/random.hpp"
	"src/shared/macros.hpp"
)

SET(SRC_FILES
	"src/device/camera.cpp"

	"src/utils/image_buffer.cpp"
	"src/utils/mesh_loader.cpp"
)

SET(INCLUDE_DIRS "include" "src")
ADD_LIBRARY(lib${PROJECT_NAME} ${HEADER_FILES} ${SRC_FILES})
TARGET_LINK_LIBRARIES(lib${PROJECT_NAME} PUBLIC
	owl::owl
	fmt::fmt
	tinyobjloader
)
TARGET_INCLUDE_DIRECTORIES(lib${PROJECT_NAME} PUBLIC ${INCLUDE_DIRS})



EMBED_PTX(
	OUTPUT_TARGET device-ptx
	PTX_LINK_LIBRARIES ${CUDA_RUNTIME_LIBRARY} lib${PROJECT_NAME}
	SOURCES "src/device/device.cu"
)


add_executable(${PROJECT_NAME} "Main.cpp")
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
	lib${PROJECT_NAME}
	device-ptx
)
