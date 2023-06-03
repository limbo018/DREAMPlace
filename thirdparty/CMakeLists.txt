#project(thirdparty)

add_subdirectory(flute)

set(OPENBLAS OFF CACHE BOOL "Turn off OPENBLAS for Limbo")
set(INSTALL_LIMBO OFF CACHE BOOL "Turn off installation for Limbo")
set(GENERATE_DOCS OFF CACHE BOOL "Turn off documentation for Limbo")
set(ENABLE_TEST OFF CACHE BOOL "Turn off test for Limbo")
add_subdirectory(Limbo)
add_subdirectory(munkres-cpp)
add_subdirectory(OpenTimer)

install(
    DIRECTORY NCTUgr.ICCAD2012 DESTINATION thirdparty
    )

