file(GLOB dirs RELATIVE ${CMAKE_CURRENT_LIST_DIR} "oc??_*")

foreach(dir ${dirs})
  add_subdirectory(${dir})
endforeach()