EMBED_PTX(
	OUTPUT_TARGET
		experiment02-ptx
	PTX_LINK_LIBRARIES
		owl::owl
	SOURCES
		"device/tracer.cu"
)

add_executable(experiment02 "hostCode.cpp")
target_link_libraries(experiment02 PRIVATE
	experiment02-ptx
	owl::owl
	fmt::fmt
	stb_image
)