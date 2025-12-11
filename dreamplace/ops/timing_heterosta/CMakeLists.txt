set(OP_NAME timing_heterosta)

set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${LIMBO_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${ZLIB_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${HETEROSTA_INCLUDE_DIR}
  )

set(LINK_LIBS utility_cxx
  # begin targets in Limbo for PlaceDB dependencies
  lefparseradapt
  defparseradapt
  verilogparser
  bookshelfparser
  gdsparser
  gzstream
  # end targets in Limbo
  OpenMP::OpenMP_CXX
  programoptions
  ${ZLIB_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  stdc++fs
  )

file(GLOB CPP_SOURCES
  src/timing_hs_cpp.cpp
  src/timing_hs_io_cpp.cpp
  src/timing_hs_pybind.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../place_io/src/PlaceDB.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../place_io/src/Node.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../place_io/src/Net.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../place_io/src/Region.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../place_io/src/Group.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../place_io/src/Params.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../place_io/src/Enums.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../place_io/src/BenchMetrics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../place_io/src/BinMap.cpp
  )

# Add CUDA sources if CUDA is enabled
if(TORCH_ENABLE_CUDA)
    file(GLOB CUDA_SOURCES
        src/timing_cuda_kernel.cu
    )
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

if(TORCH_ENABLE_CUDA)
    add_pytorch_extension(${TARGET_NAME}_cpp
        ${CPP_SOURCES} ${CUDA_SOURCES}
        EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
        EXTRA_LINK_LIBRARIES ${LINK_LIBS})
    target_compile_definitions(${TARGET_NAME}_cpp PRIVATE CUDA_FOUND)
else()
    add_pytorch_extension(${TARGET_NAME}_cpp
        ${CPP_SOURCES}
        EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
        EXTRA_LINK_LIBRARIES ${LINK_LIBS})
endif()


# Derive the library directory from the first item in HETEROSTA_LIB list
if(HETEROSTA_LIB)
    list(GET HETEROSTA_LIB 0 _HETEROSTA_ANYLIB)
    get_filename_component(_HETEROSTA_LIB_DIR "${_HETEROSTA_ANYLIB}" DIRECTORY)
    if(EXISTS "${_HETEROSTA_LIB_DIR}")
        message(STATUS "timing_heterosta: adding heterosta lib dir to linker search path: ${_HETEROSTA_LIB_DIR}")
        # New CMake prefers target_link_directories over link_directories
        if(COMMAND target_link_directories)
            target_link_directories(${TARGET_NAME}_cpp PRIVATE "${_HETEROSTA_LIB_DIR}")
        else()
            link_directories("${_HETEROSTA_LIB_DIR}")
        endif()
    endif()
endif()

# Link by name to ensure DT_NEEDED == 'libheterosta.so'
target_link_libraries(${TARGET_NAME}_cpp PRIVATE -lheterosta)

set_target_properties(${TARGET_NAME}_cpp PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    BUILD_RPATH_USE_ORIGIN TRUE)

install(FILES ${HETEROSTA_LIB}
          DESTINATION dreamplace/ops/${OP_NAME})

install(TARGETS 
  ${TARGET_NAME}_cpp 
  DESTINATION dreamplace/ops/${OP_NAME})

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
