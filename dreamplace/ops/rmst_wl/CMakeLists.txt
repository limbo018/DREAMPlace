cmake_minimum_required(VERSION 3.0.2)

project(rmst_wl)

find_program(PYTHON "python")
find_path(FLUTE_INCLUDE_DIRS flute.h PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdparty/*)
message("-- FLUTE_INCLUDE_DIRS ${FLUTE_INCLUDE_DIRS}")
get_filename_component(MATCH_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../ ABSOLUTE)
get_filename_component(REPLACE_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../../ ABSOLUTE)
string(REPLACE ${MATCH_PATH} ${REPLACE_PATH} FLUTE_LINK_DIRS ${FLUTE_INCLUDE_DIRS})
message("-- FLUTE_LINK_DIRS ${FLUTE_LINK_DIRS}")

if (PYTHON)
    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/src/setup.py.in")
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
        DEPENDS ${SOURCES} flute
        )

    add_custom_target(clean_${PROJECT_NAME}
        COMMAND ${PYTHON} ${SETUP_PY} develop --uninstall
        COMMAND rm -rf ${OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/build ${CMAKE_CURRENT_BINARY_DIR}/dist ${CMAKE_CURRENT_BINARY_DIR}/*.egg-info
        )

    add_custom_target(${PROJECT_NAME} ALL DEPENDS ${OUTPUT})

    #install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
endif()
