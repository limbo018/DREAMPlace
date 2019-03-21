cmake_minimum_required(VERSION 3.0.2)

add_subdirectory(utility)
add_subdirectory(dct)
add_subdirectory(density_overflow)
add_subdirectory(density_potential)
add_subdirectory(logsumexp_wirelength)
add_subdirectory(draw_place)
add_subdirectory(electric_potential)
add_subdirectory(greedy_legalize)
add_subdirectory(hpwl)
add_subdirectory(move_boundary)
add_subdirectory(weighted_average_wirelength)
add_subdirectory(rmst_wl)
add_subdirectory(place_io)

install(
    CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_INSTALL_PREFIX}/lib/dreamplace/ops/__init__.py)"
    )
