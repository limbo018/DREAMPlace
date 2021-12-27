set(OP_NAME fence_region)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
list(FILTER INSTALL_SRCS EXCLUDE REGEX ".*setup.py$")
install(
    FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME}
    )

