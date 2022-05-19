ADD_LIBRARY(lib${PROJECT_NAME})

TARGET_SOURCES(lib${PROJECT_NAME} PRIVATE

        "src/device/disney.hpp"

        # SRC
        "src/owl.hpp"
        "src/types.hpp"
        "src/random.hpp"
        "src/macros.hpp"
        "settings.hpp"
        "src/sample_methods.hpp"

        "src/camera.hpp"
        "src/camera.cpp"

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

        src/device/path_tracer.hpp src/device/device_global.hpp )

TARGET_LINK_LIBRARIES(lib${PROJECT_NAME} PUBLIC
        owl::owl
        fmt::fmt
        tinyobjloader)

TARGET_INCLUDE_DIRECTORIES(lib${PROJECT_NAME} PUBLIC
        "src")

EMBED_PTX(
        OUTPUT_TARGET device-ptx
        PTX_LINK_LIBRARIES ${CUDA_RUNTIME_LIBRARY} lib${PROJECT_NAME}
        SOURCES "src/device/device.cu"
)

ADD_EXECUTABLE(${PROJECT_NAME} "Main.cpp" )
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        lib${PROJECT_NAME}
        device-ptx)
