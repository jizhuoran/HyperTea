# Collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include(../cmake/Modules/FindGFlags.cmake)
if(GFLAGS_FOUND)
    include_directories(${GFLAGS_INCLUDE_DIRS})
    list(APPEND Hypertea_LINK ${GFLAGS_LIBRARIES})
endif()

# Build each source file independently
foreach(source ${srcs})
  get_filename_component(name ${source} NAME_WE)

  # hypertea target already exits
  if(name MATCHES "hypertea")
    set(name ${name}.bin)
  endif()

  # target
  add_executable(${name} ${source})
  target_link_libraries(${name} ${Hypertea_LINK})
  hypertea_default_properties(${name})

  # set back RUNTIME_OUTPUT_DIRECTORY
  hypertea_set_runtime_directory(${name} "${PROJECT_BINARY_DIR}/tools")
  hypertea_set_solution_folder(${name} tools)

  # restore output name without suffix
  if(name MATCHES "hypertea.bin")
    set_target_properties(${name} PROPERTIES OUTPUT_NAME hypertea)
  endif()

  # Install
  install(TARGETS ${name} DESTINATION bin)
endforeach(source)
