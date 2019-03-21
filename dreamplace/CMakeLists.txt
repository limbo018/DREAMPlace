cmake_minimum_required(VERSION 3.0.2)

add_subdirectory(ops)

install(
    CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_INSTALL_PREFIX}/lib/dreamplace/__init__.py)"
    )
