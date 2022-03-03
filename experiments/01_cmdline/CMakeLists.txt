EMBED_PTX(
	OUTPUT_TARGET
		experiment01-ptx
	PTX_LINK_LIBRARIES
		owl::owl
	SOURCES
		"device/tracer.cu"
)

add_executable(experiment01 "hostCode.cpp")
target_link_libraries(experiment01 PRIVATE
	# PTX file
	experiment01-ptx

	# my logger lib
	simpleLogger

	# external lib
	owl::owl
	fmt::fmt
	stb_image
)