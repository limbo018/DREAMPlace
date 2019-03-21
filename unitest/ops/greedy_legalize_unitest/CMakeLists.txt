cmake_minimum_required(VERSION 3.0.2)

project(greedy_legalize_unitest)
get_filename_component(UTILITY_LIBRARY_DIRS ${CMAKE_CURRENT_BINARY_DIR}/../../../dreamplace/ops/utility ABSOLUTE)

file(GLOB SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    )
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../../dreamplace/ops")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set (CMAKE_CXX_STANDARD 11)

add_executable(abacus_unitest ${SOURCES})
target_link_libraries(abacus_unitest ${UTILITY_LIBRARY_DIRS}/libutility.a)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
install(
    FILES ${INSTALL_SRCS} DESTINATION unitest/ops/${PROJECT_NAME}
    )
install(
    TARGETS abacus_unitest DESTINATION unitest/ops/${PROJECT_NAME}
    )
