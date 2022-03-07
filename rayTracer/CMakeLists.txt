
EMBED_PTX(
	OUTPUT_TARGET
		myDeviceCode-ptx
	PTX_LINK_LIBRARIES
		owl::owl
	SOURCES
		"deviceCode.cu"
)

set(BA_SOURCES
	"hostCode.cpp"
	"ObjLoader.cpp"
)

SET(BA_HEADERS
	"ObjLoader.hpp"
    "Renderer.hpp"
)

add_executable(bachelorThesis ${BA_SOURCES} ${BA_HEADERS})
target_link_libraries(bachelorThesis
	owl::owl
	fmt::fmt
	glm::glm
	tinyobjloader
	myDeviceCode-ptx
)

if(MSVC)
	# not receiving errors due to msvc safe functions
	add_compile_definitions(_CRT_SECURE_NO_WARNINGS=1)
endif()