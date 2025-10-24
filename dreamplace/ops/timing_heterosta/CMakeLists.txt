set(OP_NAME timing_heterosta)

set(TARGET_NAME ${OP_NAME})

# Find HeteroSTA library
find_library(HETEROSTA_LIB 
    NAMES libheterosta.so heterosta
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src
    NO_DEFAULT_PATH
    REQUIRED)

if(NOT HETEROSTA_LIB)
    message(FATAL_ERROR "HeteroSTA library not found in ${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()

set(INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${LIMBO_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${ZLIB_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
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
  ${HETEROSTA_LIB}
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

set_target_properties(${TARGET_NAME}_cpp PROPERTIES
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH "$ORIGIN")

install(FILES ${HETEROSTA_LIB}
          DESTINATION dreamplace/ops/${OP_NAME})

install(TARGETS 
  ${TARGET_NAME}_cpp 
  DESTINATION dreamplace/ops/${OP_NAME})

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
