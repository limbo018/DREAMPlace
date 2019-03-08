cmake_minimum_required(VERSION 3.0.2)

project(DREAMPlace)

#if (NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
    #endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if (NOT CMAKE_SITE_PACKAGE)
    set(CMAKE_SITE_PACKAGE "" CACHE STRING
        "Choose the python site-package of build, options are empty --user."
        FORCE)
endif(NOT CMAKE_SITE_PACKAGE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_subdirectory(thirdparty)
add_subdirectory(dreamplace)
