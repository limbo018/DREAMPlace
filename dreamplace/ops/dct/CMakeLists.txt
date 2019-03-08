cmake_minimum_required(VERSION 3.0.2)

project(dct)

find_program(PYTHON "python")
find_package(CUDA 9.0)

if (NOT CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "-gencode=arch=compute_60,code=sm_60")
endif()

if (PYTHON)
    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/src/setup.py.in")
    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/src/setup.py")
    file(GLOB SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
        )
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.stamp")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})

    add_custom_command(OUTPUT ${OUTPUT}
        COMMAND ${PYTHON} ${SETUP_PY} install ${CMAKE_SITE_PACKAGE}
        COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
        DEPENDS ${SOURCES}
        )

    add_custom_target(clean_${PROJECT_NAME}
        COMMAND ${PYTHON} ${SETUP_PY} develop --uninstall
        COMMAND rm -rf ${OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/build ${CMAKE_CURRENT_BINARY_DIR}/dist ${CMAKE_CURRENT_BINARY_DIR}/*.egg-info
        )

    add_custom_target(${PROJECT_NAME} ALL DEPENDS ${OUTPUT})

    #install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
endif()
