
ADD_LIBRARY(lib${PROJECT_NAME}

        # SRC
        "src/owl.hpp"
        "src/math.hpp"
        "src/types.hpp"
        "src/random.hpp"
        "src/macros.hpp"
        "src/sample_methods.hpp"

        "src/camera.hpp"
        "src/camera.cpp"

        # SRC/UTILS

        "src/utils/image_buffer.hpp"
        "src/utils/image_buffer.cpp"

        "src/utils/mesh_loader.hpp"
        "src/utils/mesh_loader.cpp"

        "src/utils/parser.hpp"
        "src/utils/parser.cpp"

        # SRC/DEVICE
        "src/device/materials.hpp"
        "src/device/bsdf_utils.hpp"

        # ------------------------------

        "src/device/device.hpp"
        "src/device/lights.hpp"



        "src/device/path_tracer.hpp" 
        "src/device/device_global.hpp")

TARGET_LINK_LIBRARIES(lib${PROJECT_NAME} PUBLIC
        owl::owl
        fmt::fmt
        cxxopts
        nlohmann_json
        tinyobjloader)

TARGET_INCLUDE_DIRECTORIES(lib${PROJECT_NAME} PUBLIC "src")

EMBED_PTX(OUTPUT_TARGET device-ptx
          PTX_LINK_LIBRARIES ${CUDA_RUNTIME_LIBRARY} lib${PROJECT_NAME}
          SOURCES "src/device/device.cu")

ADD_EXECUTABLE(${PROJECT_NAME} "Main.cpp")
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        lib${PROJECT_NAME}
        device-ptx)