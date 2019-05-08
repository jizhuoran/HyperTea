# --[ Hypertea library


file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)


if(ANDROID)

file(GLOB_RECURSE jni-srcs ${CMAKE_CURRENT_SOURCE_DIR}/../android/*.cpp)
add_library(hypertea STATIC ${srcs} ${jni-srcs})
target_link_libraries(hypertea ${Hypertea_LINKER_LIBS} log)

else()

add_library(hypertea STATIC ${srcs})
target_link_libraries(hypertea ${Hypertea_LINKER_LIBS})

endif()

#hypertea_default_properties(hypertea)

#INSTALL(TARGETS hypertea DESTINATION lib)
