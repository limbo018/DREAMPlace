set(OP_NAME global_swap)

if (CUDA_FOUND)
    file(GLOB CUDA_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
        )
    set(CUDA_LINKED ${OP_NAME}_cuda_linked)
    # try to turn on -fPIC for cuda, but not really working
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CUDA_SEPARABLE_COMPILATION TRUE)
    list(APPEND CUDA_NVCC_FLAGS
        -O3 -std=c++11 ${CMAKE_CUDA_FLAGS} 
        )
    list(APPEND CUDA_NVCC_FLAGS --compiler-options -fPIC)

    cuda_add_library(
        ${CUDA_LINKED} STATIC 
        ${CUDA_SOURCES}
        )

    target_include_directories(
        ${CUDA_LINKED} PUBLIC 
        ${OPS_DIR}
		${LIMBO_SOURCE_DIR}
        ${CUB_DIR}
        ${Boost_INCLUDE_DIRS}
        )

    # Request that cuda_linked be built with -std=c++11
    # As this is a public compile feature anything that links to cuda_linked
    # will also build with -std=c++11
	#target_compile_features(${CUDA_LINKED} PUBLIC cxx_std_11)

    # We need to explicitly state that we need all CUDA files in the cuda_linked
    # library to be built with -dc as the member functions could be called by
    # other libraries and executables
    set_target_properties( ${CUDA_LINKED}
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )
endif(CUDA_FOUND)

if (PYTHON)
    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    file(GLOB SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
        )
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/${OP_NAME}.stamp")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})

    if (CUDA_FOUND)
        add_custom_command(OUTPUT ${OUTPUT}
            COMMAND ${PYTHON} ${SETUP_PY} build --build-temp=${CMAKE_CURRENT_BINARY_DIR}/build --build-lib=${CMAKE_CURRENT_BINARY_DIR}/lib
            COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
            DEPENDS ${SOURCES} ${CUDA_LINKED} gdsparser gzstream
            )
    else(CUDA_FOUND)
        add_custom_command(OUTPUT ${OUTPUT}
            COMMAND ${PYTHON} ${SETUP_PY} build --build-temp=${CMAKE_CURRENT_BINARY_DIR}/build --build-lib=${CMAKE_CURRENT_BINARY_DIR}/lib
            COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
            DEPENDS ${SOURCES} gdsparser gzstream
            )
    endif(CUDA_FOUND)

    add_custom_target(clean_${OP_NAME}
        COMMAND rm -rf ${OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/build ${CMAKE_CURRENT_BINARY_DIR}/lib
        )

    add_custom_target(${OP_NAME} ALL DEPENDS ${OUTPUT})

    install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/ DESTINATION dreamplace/ops/${OP_NAME}
        )
    file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
    list(FILTER INSTALL_SRCS EXCLUDE REGEX ".*setup.py$")
    install(
        FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME}
        )
endif(PYTHON)
