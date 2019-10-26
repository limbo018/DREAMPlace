cmake_minimum_required(VERSION 3.8.2)

project(DREAMPlace)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Prefix prepended to install directories" FORCE )
endif()
message("-- CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message("-- CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

if(NOT CMAKE_CXX_ABI)
    set(CMAKE_CXX_ABI 0 CACHE STRING
        "Choose the value for _GLIBCXX_USE_CXX11_ABI, options are: 0|1."
        FORCE)
endif(NOT CMAKE_CXX_ABI)
message("-- CMAKE_CXX_ABI: _GLIBCXX_USE_CXX11_ABI=${CMAKE_CXX_ABI}")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${CMAKE_CXX_ABI})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 11)

find_program(PYTHON "python" REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Boost 1.55.0 REQUIRED)
get_filename_component(Boost_DIR ${Boost_INCLUDE_DIRS}/../ ABSOLUTE)
find_package(CUDA 9.0)
find_package(Cairo)
message("-- Cairo: ${CAIRO_INCLUDE_DIRS}")

get_filename_component(OPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dreamplace/ops ABSOLUTE)
get_filename_component(UTILITY_LIBRARY_DIRS ${CMAKE_CURRENT_BINARY_DIR}/dreamplace/ops/utility ABSOLUTE)
message("-- OPS_DIR ${OPS_DIR}")
message("-- UTILITY_LIBRARY_DIRS ${UTILITY_LIBRARY_DIRS}")

# thirdparty libraries 
# flute for steiner tree generation 
find_path(FLUTE_INCLUDE_DIRS flute.h PATHS ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/*)
string(REPLACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} FLUTE_LINK_DIRS ${FLUTE_INCLUDE_DIRS})
message("-- FLUTE_INCLUDE_DIRS ${FLUTE_INCLUDE_DIRS}")
message("-- FLUTE_LINK_DIRS ${FLUTE_LINK_DIRS}")

# Limbo for parsers 
set(LIMBO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Limbo)
message("-- LIMBO_SOURCE_DIR ${LIMBO_SOURCE_DIR}")
set(LIMBO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/Limbo)
message("-- LIMBO_BINARY_DIR ${LIMBO_BINARY_DIR}")
set(CUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub)
message("-- CUB_DIR ${CUB_DIR}")

# munkres-cpp for Hungarian algorithm 
set(MUNKRES_CPP_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/munkres-cpp/src)
set(MUNKRES_CPP_LINK_DIRS ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/munkres-cpp)
message("-- MUNKRES_CPP_INCLUDE_DIRS ${MUNKRES_CPP_INCLUDE_DIRS}")
message("-- MUNKRES_CPP_LINK_DIRS ${MUNKRES_CPP_LINK_DIRS}")

# lemon for network flow algorithms 
find_path(LEMON_SOURCE_DIR lemon PATHS ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Limbo/limbo/thirdparty/* 
	NO_DEFAULT_PATH
	)
set(LEMON_INCLUDE_DIRS "${LEMON_SOURCE_DIR}")
string(REPLACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} LEMON_BINARY_DIR ${LEMON_SOURCE_DIR})
list(APPEND LEMON_INCLUDE_DIRS "${LEMON_BINARY_DIR}")
set(LEMON_LINK_DIRS ${LEMON_BINARY_DIR}/lemon)
message("-- LEMON_INCLUDE_DIRS ${LEMON_INCLUDE_DIRS}")
message("-- LEMON_LINK_DIRS ${LEMON_LINK_DIRS}")

if (NOT CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "-arch=sm_60"
        "-gencode=arch=compute_60,code=sm_60"
        "-gencode=arch=compute_61,code=sm_61"
        "-gencode=arch=compute_70,code=sm_70"
        )
    if (${CUDA_VERSION_MAJOR} VERSION_GREATER "9")
        list(APPEND CMAKE_CUDA_FLAGS 
            "-gencode=arch=compute_75,code=sm_75"
            "-gencode=arch=compute_75,code=compute_75"
            )
    endif(${CUDA_VERSION_MAJOR} VERSION_GREATER "9")
endif(NOT CMAKE_CUDA_FLAGS)
message("-- CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

add_subdirectory(thirdparty)
add_subdirectory(dreamplace)
add_subdirectory(unitest)
add_subdirectory(benchmarks)
add_subdirectory(test)

install(
    CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_INSTALL_PREFIX}/__init__.py)"
    )

