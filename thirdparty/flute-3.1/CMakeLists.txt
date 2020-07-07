cmake_minimum_required(VERSION 2.8.12)

project(flute)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(SOURCES 
${CMAKE_CURRENT_SOURCE_DIR}/dist.c 
${CMAKE_CURRENT_SOURCE_DIR}/dl.c 
${CMAKE_CURRENT_SOURCE_DIR}/err.c 
${CMAKE_CURRENT_SOURCE_DIR}/heap.c 
${CMAKE_CURRENT_SOURCE_DIR}/mst2.c 
${CMAKE_CURRENT_SOURCE_DIR}/neighbors.c 
${CMAKE_CURRENT_SOURCE_DIR}/bookshelf_IO.c 
${CMAKE_CURRENT_SOURCE_DIR}/memAlloc.c 
${CMAKE_CURRENT_SOURCE_DIR}/flute.c 
${CMAKE_CURRENT_SOURCE_DIR}/flute_mst.c)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(${PROJECT_NAME} STATIC ${SOURCES})
add_executable(flute-net flute-net.c)
target_link_libraries(flute-net ${PROJECT_NAME} m)
add_executable(flute-ckt flute-ckt.c)
target_link_libraries(flute-ckt ${PROJECT_NAME} m)
add_executable(rand-pts rand-pts.c)
target_link_libraries(rand-pts ${PROJECT_NAME})

#install(TARGETS ${PROJECT_NAME} 
#    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
#    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
#    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

#file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.dat")
#install(
#    FILES ${INSTALL_SRCS} DESTINATION thirdparty/${PROJECT_NAME}
#    )
