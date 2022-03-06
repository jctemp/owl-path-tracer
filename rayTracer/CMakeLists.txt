
EMBED_PTX(
	OUTPUT_TARGET
		myDeviceCode-ptx
	PTX_LINK_LIBRARIES
		owl::owl
	SOURCES
		"deviceCode.cu"
)


add_executable(bachelorThesis "hostCode.cpp" "utils.hpp")
target_link_libraries(bachelorThesis
	owl::owl
	fmt::fmt
	tinyobjloader
	myDeviceCode-ptx
)

if(MSVC)
	add_compile_definitions(_CRT_SECURE_NO_WARNINGS=1)
endif()