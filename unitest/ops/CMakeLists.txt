add_subdirectory(place_io_unitest)
add_subdirectory(greedy_legalize_unitest)
add_subdirectory(global_swap_unitest)
add_subdirectory(independent_set_matching_unitest)
add_subdirectory(k_reorder_unitest)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
install(
    FILES ${INSTALL_SRCS} DESTINATION unitest/ops
    )
