cmake_minimum_required(VERSION 3.0.2)

#project(thirdparty)

add_subdirectory(flute-3.1)

set(OPENBLAS OFF CACHE BOOL "Turn off OPENBLAS for Limbo")
set(INSTALL_LIMBO OFF CACHE BOOL "Turn off installation for Limbo")
add_subdirectory(Limbo)
