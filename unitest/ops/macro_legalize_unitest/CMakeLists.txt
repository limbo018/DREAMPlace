cmake_minimum_required(VERSION 3.0.2)

project(macro_legalize_unitest)
get_filename_component(UTILITY_LIBRARY_DIRS ${CMAKE_CURRENT_BINARY_DIR}/../../../dreamplace/ops/utility ABSOLUTE)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
install(
    FILES ${INSTALL_SRCS} DESTINATION unitest/ops/${PROJECT_NAME}
    )
