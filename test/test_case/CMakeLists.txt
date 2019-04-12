# The option allows to include in build only selected test files and exclude all others
# Usage example:
#  cmake -DBUILD_only_tests="common,net,blob,im2col_kernel"


file(GLOB test_srcs    ${CMAKE_SOURCE_DIR}/test/test_case/test_*.cpp)


set(BUILD_only_tests "" CACHE STRING "Blank or comma-separated list of test files to build without 'test_' prefix and extension")
hypertea_leave_only_selected_tests(test_srcs ${BUILD_only_tests})


set(the_target test.testbin)
set(test_args --gtest_shuffle)

# ---[ Adding test target
include_directories(${CMAKE_SOURCE_DIR}/test)

add_executable(${the_target} EXCLUDE_FROM_ALL ${test_srcs} 
											  ${CMAKE_SOURCE_DIR}/test/gtest/gtest-all.cpp 
											  ${CMAKE_SOURCE_DIR}/test/gtest/gtest_main.cc)

target_link_libraries(${the_target} ${Hypertea_LINK} pthread)

hypertea_default_properties(${the_target})
hypertea_set_runtime_directory(${the_target} "${PROJECT_BINARY_DIR}/test")


# ---[ Adding runtest
add_custom_target(runtest COMMAND ${the_target} ${test_args}
                          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
