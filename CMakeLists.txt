cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0046)
    cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

set(HYPERTEA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${HYPERTEA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)


option(USE_OPENCL "USE OPENCL SUPPORT" ON)

add_definitions(-DUSE_NEON_MATH)

 
include(./cmake/Utils.cmake)
include(./cmake/Misc.cmake)
include(./cmake/Targets.cmake)

# Set CXX_FLAGS for different platform

SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++17")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++17")


if(NOT MSVC)
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-sign-compare")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-sign-compare")
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++17")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++17")
    if(IOS)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fembed-bitcode")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fembed-bitcode")
    elseif(ANDROID)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fembed-bitcode")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fembed-bitcode")
        
    endif()
endif()

# Set search path for cross compile
if(ANDROID)
    STRING(REGEX MATCH "[^ ]+" ANDROID_ABI_PREFIX ${ANDROID_ABI})
    set(THIRD_PREFIX ${ANDROID_ABI_PREFIX}-${ANDROID_NATIVE_API_LEVEL}-)
    list(INSERT CMAKE_FIND_ROOT_PATH 0 ${CMAKE_SOURCE_DIR}/third_party/${THIRD_PREFIX}protobuf ${CMAKE_SOURCE_DIR}/third_party/${THIRD_PREFIX}OpenBLAS ${CMAKE_SOURCE_DIR}/third_party/${THIRD_PREFIX}CLBlast ${OPENCL_ROOT})
elseif(IOS)
    list(INSERT CMAKE_FIND_ROOT_PATH 0 ${CMAKE_SOURCE_DIR}/third_party/ios-protobuf)
endif()
#message(STATUS "Change CMAKE_FIND_ROOT_PATH to: ${CMAKE_FIND_ROOT_PATH}")

include(./cmake/Modules/FindOpenBLAS.cmake)
include(./cmake/Modules/FindvecLib.cmake)
include(./cmake/Modules/FindAtlas.cmake)


# Select blas lib for different platform
# APPLE only, 1st
if(VECLIB_FOUND)
    include_directories(${vecLib_INCLUDE_DIR})
    list(APPEND Hypertea_LINKER_LIBS ${vecLib_LINKER_LIBS})
elseif(OpenBLAS_FOUND)
    include_directories(${OpenBLAS_INCLUDE_DIR})
    list(APPEND Hypertea_LINKER_LIBS ${OpenBLAS_LIB})
elseif(ATLAS_FOUND)
    include_directories(${Atlas_CBLAS_INCLUDE_DIR})
    list(APPEND Hypertea_LINKER_LIBS ${Atlas_CBLAS_LIBRARY})
else()
    MESSAGE(FATAL_ERROR "BLAS (VecLib/OpenBLAS/Atlas) library not found.")
endif()



if(USE_OPENCL)

add_definitions(-DUSE_OPENCL)

include(./cmake/Modules/FindOpenCL.cmake)
include_directories(${OPENCL_INCLUDE_DIRS})
list(APPEND Hypertea_LINKER_LIBS ${OPENCL_LIBRARIES})  

include(./cmake/Modules/FindCLBlast.cmake)
include_directories(${CLBlast_INCLUDE_DIR})
list(APPEND Hypertea_LINKER_LIBS ${CLBlast_LIB})


endif()

hypertea_set_hypertea_link()

add_subdirectory(src/hypertea)

if(NOT ANDROID)
    add_subdirectory(tools)
endif()

