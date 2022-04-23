add_library(simpleLogger INTERFACE)
target_sources(simpleLogger INTERFACE 
	"${CMAKE_CURRENT_SOURCE_DIR}/SimpleLogger.hpp"
)