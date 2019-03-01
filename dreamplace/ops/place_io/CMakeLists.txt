cmake_minimum_required(VERSION 2.8.12)

project(draw_place)

find_program(PYTHON "python")
find_package(CUDA 9.0 REQUIRED)
find_package(Boost 1.55.0 COMPONENTS iostreams timer chrono system)
find_package(ZLIB REQUIRED)
find_package(Cairo)
get_filename_component(LIMBO_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../thirdparty/Limbo ABSOLUTE)
get_filename_component(OPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
get_filename_component(UTILITY_LIBRARY_DIRS ${CMAKE_CURRENT_BINARY_DIR}/../utility ABSOLUTE)

if (NOT CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "-gencode=arch=compute_60,code=sm_60")
endif()

if (PYTHON)
    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/src/setup.py")
    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/src/setup.py")
    file(GLOB SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
        )
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.stamp")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})

    add_custom_command(OUTPUT ${OUTPUT}
        COMMAND ${PYTHON} ${SETUP_PY} install ${CMAKE_SITE_PACKAGE}
        COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
        DEPENDS ${SOURCES}
        )

    add_custom_target(extraclean
        COMMAND ${PYTHON} ${SETUP_PY} develop --uninstall
        COMMAND rm -rf ${OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/build ${CMAKE_CURRENT_BINARY_DIR}/dist ${CMAKE_CURRENT_BINARY_DIR}/*.egg-info
        )

    add_custom_target(${PROJECT_NAME} ALL DEPENDS ${OUTPUT})

    #install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
endif()
