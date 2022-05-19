IF (MSVC)
    # not receiving errors due to msvc safe functions
    ADD_COMPILE_DEFINITIONS(_CRT_SECURE_NO_WARNINGS=1)
ENDIF ()

SET(FILES
        # SRC
        "src/owl.hpp"
        "src/types.hpp"
        "src/random.hpp"

        "src/camera.hpp"
        "src/camera.cpp"

        src/sample_methods.hpp

        # SRC/UTILS

        "src/utils/image_buffer.hpp"
        "src/utils/image_buffer.cpp"

        "src/utils/mesh_loader.hpp"
        "src/utils/mesh_loader.cpp"

        # ------------------------------

        "src/device/device.hpp"

        "src/math.hpp"
        "src/device/materials/Lambert.hpp"
        "src/device/materials/Disney.hpp"
        "src/device/materials/BrdfUtils.hpp"

        "src/shared/shared.hpp"
        src/device/path_tracer.hpp )


SET(INCLUDE_DIRS "include" "src")
ADD_LIBRARY(lib${PROJECT_NAME} ${FILES})
TARGET_LINK_LIBRARIES(lib${PROJECT_NAME} PUBLIC
        owl::owl
        fmt::fmt
        tinyobjloader)

TARGET_INCLUDE_DIRECTORIES(lib${PROJECT_NAME} PUBLIC ${INCLUDE_DIRS})

EMBED_PTX(
        OUTPUT_TARGET device-ptx
        PTX_LINK_LIBRARIES ${CUDA_RUNTIME_LIBRARY} lib${PROJECT_NAME}
        SOURCES "src/device/device.cu"
)


ADD_EXECUTABLE(${PROJECT_NAME} "Main.cpp")
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        lib${PROJECT_NAME}
        device-ptx)
