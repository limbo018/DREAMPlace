set(OP_NAME timing)

set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${FLUTE_INCLUDE_DIRS}
  ${OPENTIMER_SOURCE_DIR}
  ${LIMBO_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
  )

set(LINK_LIBS utility_cxx
  OpenMP::OpenMP_CXX
  programoptions
  flute
  OpenTimer
  stdc++fs
  )

file(GLOB CPP_SOURCES
  src/timing_cpp.cpp
  src/timing_io_cpp.cpp
  src/timing_pybind.cpp
  )
add_pytorch_extension(${TARGET_NAME}_cpp 
  ${CPP_SOURCES}
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(TARGETS 
  ${TARGET_NAME}_cpp 
  DESTINATION dreamplace/ops/${OP_NAME})

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})

