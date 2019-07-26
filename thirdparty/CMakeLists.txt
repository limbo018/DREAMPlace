#project(thirdparty)

add_subdirectory(flute-3.1)

set(OPENBLAS OFF CACHE BOOL "Turn off OPENBLAS for Limbo")
set(INSTALL_LIMBO OFF CACHE BOOL "Turn off installation for Limbo")
set(GENERATE_DOCS OFF CACHE BOOL "Turn off documentations")
add_subdirectory(Limbo)
add_subdirectory(munkres-cpp)
