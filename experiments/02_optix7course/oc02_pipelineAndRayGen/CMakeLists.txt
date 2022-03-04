EMBED_PTX(
	OUTPUT_TARGET
		oc02_pipelineAndRayGen-ptx
	PTX_LINK_LIBRARIES
		owl::owl
	SOURCES
		"deviceCode.cu"
)

add_executable(oc02_pipelineAndRayGen
	# headers
	"LaunchParams.hpp"
	
	# sources
	"main.cpp"
)
target_link_libraries(oc02_pipelineAndRayGen
	oc02_pipelineAndRayGen-ptx
	owl::owl
)