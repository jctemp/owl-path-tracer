EMBED_PTX(
	OUTPUT_TARGET
		experiment03-ptx
	PTX_LINK_LIBRARIES
		owl::owl
	SOURCES
		"device/tracer.cu"
)

add_executable(experiment03 "hostCode.cpp")
target_link_libraries(experiment03 PRIVATE
	experiment03-ptx
	owl::owl
	fmt::fmt
	stb_image
)