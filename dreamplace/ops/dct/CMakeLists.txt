set(OP_NAME dct)

set(INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

set(LINK_LIBS utility_cxx
  OpenMP::OpenMP_CXX
  )

add_pytorch_extension(dct_cpp 
  src/dct.cpp
  src/dst.cpp
  src/dxt.cpp
  src/dct_2N.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

add_pytorch_extension(dct_lee_cpp
  src/dct_lee.cpp 
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

add_pytorch_extension(dct2_fft2_cpp
  src/dct2_fft2.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(TARGETS 
  dct_cpp dct_lee_cpp dct2_fft2_cpp 
  DESTINATION dreamplace/ops/${OP_NAME})

if(TORCH_ENABLE_CUDA)

# shared by dct_cuda and dct_lee_cuda 
add_torch_extension(dct_dst_kernel_cuda
  src/dct_cuda_kernel.cu
  src/dst_cuda_kernel.cu
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

add_pytorch_extension(dct_cuda 
  src/dct_cuda.cpp
  src/dst_cuda.cpp
  src/dxt_cuda.cpp
  src/dct_2N_cuda.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES dct_dst_kernel_cuda ${LINK_LIBS})

add_pytorch_extension(dct_lee_cuda
  src/dct_lee_cuda.cpp
  src/dct_lee_cuda_kernel.cu
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES dct_dst_kernel_cuda ${LINK_LIBS})

add_pytorch_extension(dct2_fft2_cuda
  src/dct2_fft2_cuda.cpp
  src/dct2_fft2_cuda_kernel.cu
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(TARGETS 
  dct_cuda dct_lee_cuda dct2_fft2_cuda 
  DESTINATION dreamplace/ops/${OP_NAME})
endif(TORCH_ENABLE_CUDA)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
