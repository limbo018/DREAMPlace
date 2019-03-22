cmake_minimum_required(VERSION 3.0.2)

add_subdirectory(place_io_unitest)
add_subdirectory(greedy_legalize_unitest)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
install(
    FILES ${INSTALL_SRCS} DESTINATION unitest/ops
    )
