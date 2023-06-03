add_subdirectory(place_io_unittest)
add_subdirectory(macro_legalize_unittest)
add_subdirectory(greedy_legalize_unittest)
add_subdirectory(abacus_legalize_unittest)
add_subdirectory(global_swap_unittest)
add_subdirectory(independent_set_matching_unittest)
add_subdirectory(k_reorder_unittest)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
install(
    FILES ${INSTALL_SRCS} DESTINATION unittest/ops
    )
