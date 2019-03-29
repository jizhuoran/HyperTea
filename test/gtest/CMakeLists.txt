add_library(gtest STATIC EXCLUDE_FROM_ALL gtest.h gtest-all.cpp)
include_directories(${CMAKE_SOURCE_DIR}/test)
hypertea_default_properties(gtest)
target_compile_definitions(gtest PUBLIC -DGTEST_USE_OWN_TR1_TUPLE)

list(APPEND ${Hypertea_LINK} gtest)


