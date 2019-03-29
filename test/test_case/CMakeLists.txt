# The option allows to include in build only selected test files and exclude all others
# Usage example:
#  cmake -DBUILD_only_tests="common,net,blob,im2col_kernel"


#add_library(gtest STATIC ${CMAKE_SOURCE_DIR}/test/gtest/gtest.h ${CMAKE_SOURCE_DIR}/test/gtest/gtest-all.cpp)
#hypertea_default_properties(gtest)
#target_include_directories(gtest PUBLIC ${Caffe_SRC_DIR})
#target_compile_definitions(gtest PUBLIC -DGTEST_USE_OWN_TR1_TUPLE)


file(GLOB test_hdrs    ${CMAKE_SOURCE_DIR}/test/test_result/*_result.h*)
file(GLOB test_srcs    ${CMAKE_SOURCE_DIR}/test/test_case/test_*.cpp)


set(BUILD_only_tests "" CACHE STRING "Blank or comma-separated list of test files to build without 'test_' prefix and extension")
hypertea_leave_only_selected_tests(test_srcs ${BUILD_only_tests})

message(INFO^_^ ${test_srcs})


# For 'make runtest' target we don't need to embed test data paths to
# source files, because test target is executed in source directory
# That's why the lines below are commented. TODO: remove them

# definition needed to include CMake generated files

# generates test_data/sample_data_list.txt.gen.cmake
#caffe_configure_testdatafile(test_data/sample_data_list.txt)

set(the_target test.testbin)
set(test_args --gtest_shuffle)

# ---[ Adding test target
include_directories(${CMAKE_SOURCE_DIR}/test)
add_executable(${the_target} EXCLUDE_FROM_ALL ${test_srcs} ${CMAKE_SOURCE_DIR}/test/gtest/gtest.h ${CMAKE_SOURCE_DIR}/test/gtest/gtest-all.cpp ${CMAKE_SOURCE_DIR}/test/gtest/gtest_main.cc)
target_link_libraries(${the_target} ${Hypertea_LINK} pthread)
hypertea_default_properties(${the_target})
hypertea_set_runtime_directory(${the_target} "${PROJECT_BINARY_DIR}/test")

message(INFO^_^ ${PROJECT_SOURCE_DIR} ${the_target} ${test_args})

# ---[ Adding runtest
add_custom_target(runtest COMMAND ${the_target} ${test_args}
                          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
