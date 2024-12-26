add_subdirectory(utility)
add_subdirectory(place_io)
add_subdirectory(dct)
add_subdirectory(pin_pos)
add_subdirectory(move_boundary)
add_subdirectory(draw_place)
add_subdirectory(density_map)
add_subdirectory(density_overflow)
add_subdirectory(density_potential)
add_subdirectory(electric_potential)
add_subdirectory(hpwl)
# add_subdirectory(rmst_wl)
add_subdirectory(weighted_average_wirelength)
add_subdirectory(logsumexp_wirelength)
add_subdirectory(fence_region)
# legalization operators
add_subdirectory(legality_check)
add_subdirectory(macro_legalize)
add_subdirectory(greedy_legalize)
add_subdirectory(abacus_legalize)
# detailed placement operators
add_subdirectory(global_swap)
add_subdirectory(independent_set_matching)
add_subdirectory(k_reorder)
# routing congestion operator
add_subdirectory(pin_utilization)
add_subdirectory(rudy)
add_subdirectory(pinrudy)
add_subdirectory(nctugr_binary)
add_subdirectory(adjust_node_area)
# timing operator
add_subdirectory(pin_weight_sum)
add_subdirectory(timing)
# gift initial placement 
add_subdirectory(gift_init)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
install(
    FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops
    )
