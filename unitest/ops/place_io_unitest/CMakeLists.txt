cmake_minimum_required(VERSION 3.0.2)

project(place_io_unitest)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
install(
    FILES ${INSTALL_SRCS} DESTINATION unitest/ops/${PROJECT_NAME}
    )
install(
    DIRECTORY simple DESTINATION unitest/ops/${PROJECT_NAME}
    )
install(
    FILES simple.golden DESTINATION unitest/ops/${PROJECT_NAME}
    )
