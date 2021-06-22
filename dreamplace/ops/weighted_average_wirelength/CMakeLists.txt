set(OP_NAME weighted_average_wirelength)

set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

set(LINK_LIBS utility_cxx
  OpenMP::OpenMP_CXX
  )

add_pytorch_extension(${TARGET_NAME}_cpp
  src/weighted_average_wirelength.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

add_pytorch_extension(${TARGET_NAME}_cpp_atomic
  src/weighted_average_wirelength_atomic.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

add_pytorch_extension(${TARGET_NAME}_cpp_merged
  src/weighted_average_wirelength_merged.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(TARGETS 
  ${TARGET_NAME}_cpp 
  ${TARGET_NAME}_cpp_atomic 
  ${TARGET_NAME}_cpp_merged 
  DESTINATION dreamplace/ops/${OP_NAME})

if(TORCH_ENABLE_CUDA)

add_torch_extension(integrate_net_weights_cuda
  src/integrate_net_weights_cuda_kernel.cu
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

add_pytorch_extension(${TARGET_NAME}_cuda
  src/weighted_average_wirelength_cuda.cpp
  src/weighted_average_wirelength_cuda_kernel.cu
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES integrate_net_weights_cuda ${LINK_LIBS})

add_pytorch_extension(${TARGET_NAME}_cuda_atomic 
  src/weighted_average_wirelength_cuda_atomic.cpp
  src/weighted_average_wirelength_cuda_atomic_kernel.cu
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES integrate_net_weights_cuda ${LINK_LIBS})

add_pytorch_extension(${TARGET_NAME}_cuda_merged
  src/weighted_average_wirelength_cuda_merged.cpp
  src/weighted_average_wirelength_cuda_merged_kernel.cu
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES integrate_net_weights_cuda ${LINK_LIBS})

install(TARGETS 
  ${TARGET_NAME}_cuda
  ${TARGET_NAME}_cuda_atomic
  ${TARGET_NAME}_cuda_merged 
  DESTINATION dreamplace/ops/${OP_NAME})
endif(TORCH_ENABLE_CUDA)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
