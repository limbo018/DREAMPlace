set(OP_NAME independent_set_matching)

set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${LIMBO_SOURCE_DIR}
  ${MUNKRES_CPP_INCLUDE_DIRS}
  ${LEMON_INCLUDE_DIRS}
  ${CUB_DIR}
  )

set(LINK_LIBS 
  #draw_place_cxx
  utility_cxx
  ${LIMBO_BINARY_DIR}/limbo/parsers/gdsii/stream/libgdsparser.a
  ${LIMBO_BINARY_DIR}/limbo/thirdparty/gzstream/libgzstream.a
  munkres
  lemon
  OpenMP::OpenMP_CXX
  ${ZLIB_LIBRARIES}
  )

if(CAIRO_FOUND)
  set(INCLUDE_DIRS ${INCLUDE_DIRS} ${CAIRO_INCLUDE_DIRS})
  set(LINK_LIBS ${LINK_LIBS} ${CAIRO_LIBRARIES})
  set(DRAWPLACE 1)
else()
  set(DRAWPLACE 0)
endif()

add_pytorch_extension(${TARGET_NAME}_cpp 
  src/independent_set_matching.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS}
  EXTRA_DEFINITIONS DRAWPLACE=${DRAWPLACE})

add_pytorch_extension(${TARGET_NAME}_sequential_cpp
  src/independent_set_matching_sequential.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS}
  EXTRA_DEFINITIONS DRAWPLACE=${DRAWPLACE})

install(TARGETS 
  ${TARGET_NAME}_cpp ${TARGET_NAME}_sequential_cpp 
  DESTINATION dreamplace/ops/${OP_NAME})

if(TORCH_ENABLE_CUDA)
set(CUDA_SEPARABLE_COMPILATION ON)
add_pytorch_extension(${TARGET_NAME}_cuda 
  src/independent_set_matching_cuda.cpp
  src/independent_set_matching_cuda_kernel.cu
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS} ${CUDA_curand_LIBRARY}
  EXTRA_DEFINITIONS DRAWPLACE=${DRAWPLACE})
install(TARGETS 
  ${TARGET_NAME}_cuda 
  DESTINATION dreamplace/ops/${OP_NAME})
set(CUDA_SEPARABLE_COMPILATION OFF)
endif(TORCH_ENABLE_CUDA)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
