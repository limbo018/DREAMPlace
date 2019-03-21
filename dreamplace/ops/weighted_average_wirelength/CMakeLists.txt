cmake_minimum_required(VERSION 2.8.12)

project(weighted_average_wirelength)

find_program(PYTHON "python")
find_package(CUDA 9.0)
get_filename_component(OPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
get_filename_component(UTILITY_LIBRARY_DIRS ${CMAKE_CURRENT_BINARY_DIR}/../utility ABSOLUTE)

if (NOT CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "-gencode=arch=compute_60,code=sm_60")
endif()

if (PYTHON)
    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    file(GLOB SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
        )
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.stamp")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})

    add_custom_command(OUTPUT ${OUTPUT}
        COMMAND ${PYTHON} ${SETUP_PY} build --build-temp=${CMAKE_CURRENT_BINARY_DIR}/build --build-lib=${CMAKE_CURRENT_BINARY_DIR}/lib
        COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
        DEPENDS ${SOURCES}
        )

    add_custom_target(clean_${PROJECT_NAME}
        COMMAND rm -rf ${OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/build ${CMAKE_CURRENT_BINARY_DIR}/lib
        )

    add_custom_target(${PROJECT_NAME} ALL DEPENDS ${OUTPUT})

    install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/ DESTINATION lib/dreamplace/ops/${PROJECT_NAME}
        )
    install(
        CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_INSTALL_PREFIX}/lib/dreamplace/ops/${PROJECT_NAME}/__init__.py)"
        )
endif()
