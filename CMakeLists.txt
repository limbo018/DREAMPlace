cmake_minimum_required(VERSION 3.8.2)

project(DREAMPlace)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Prefix prepended to install directories")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message("-- CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

if (NOT CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "-gencode=arch=compute_60,code=sm_60")
endif()
message("-- CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

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
find_package(Boost 1.62.0 EQUIRED)
get_filename_component(Boost_DIR ${Boost_INCLUDE_DIRS}/../ ABSOLUTE)
find_package(CUDA 9.0)
find_package(Cairo)

get_filename_component(OPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dreamplace/ops ABSOLUTE)
get_filename_component(UTILITY_LIBRARY_DIRS ${CMAKE_CURRENT_BINARY_DIR}/dreamplace/ops/utility ABSOLUTE)
message("-- OPS_DIR ${OPS_DIR}")
message("-- UTILITY_LIBRARY_DIRS ${UTILITY_LIBRARY_DIRS}")

find_path(FLUTE_INCLUDE_DIRS flute.h PATHS ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/*)
string(REPLACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} FLUTE_LINK_DIRS ${FLUTE_INCLUDE_DIRS})
message("-- FLUTE_INCLUDE_DIRS ${FLUTE_INCLUDE_DIRS}")
message("-- FLUTE_LINK_DIRS ${FLUTE_LINK_DIRS}")

set(LIMBO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Limbo)
message("-- LIMBO_SOURCE_DIR ${LIMBO_SOURCE_DIR}")
set(LIMBO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/Limbo)
message("-- LIMBO_BINARY_DIR ${LIMBO_BINARY_DIR}")

add_subdirectory(thirdparty)
add_subdirectory(dreamplace)
add_subdirectory(unitest)
add_subdirectory(benchmarks)
add_subdirectory(test)

install(
    CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_INSTALL_PREFIX}/__init__.py)"
    )
