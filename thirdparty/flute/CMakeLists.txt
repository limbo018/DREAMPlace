cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(flute)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release."
    FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/flute.cpp)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(${PROJECT_NAME} STATIC ${SOURCES})

install(TARGETS ${PROJECT_NAME} DESTINATION thirdparty/${PROJECT_NAME})
install(DIRECTORY lut.ICCAD2015 DESTINATION thirdparty/${PROJECT_NAME})
