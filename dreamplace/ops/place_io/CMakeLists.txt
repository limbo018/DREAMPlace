set(OP_NAME place_io)

set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${LIMBO_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${ZLIB_INCLUDE_DIRS}
  )

set(LINK_LIBS 
  #draw_place_cxx
  utility_cxx
  ${LIMBO_BINARY_DIR}/limbo/parsers/lef/adapt/liblefparseradapt.a
  ${LIMBO_BINARY_DIR}/limbo/parsers/def/adapt/libdefparseradapt.a
  ${LIMBO_BINARY_DIR}/limbo/parsers/verilog/bison/libverilogparser.a
  ${LIMBO_BINARY_DIR}/limbo/parsers/bookshelf/bison/libbookshelfparser.a
  ${LIMBO_BINARY_DIR}/limbo/parsers/gdsii/stream/libgdsparser.a
  ${LIMBO_BINARY_DIR}/limbo/programoptions/libprogramoptions.a
  ${LIMBO_BINARY_DIR}/limbo/thirdparty/gzstream/libgzstream.a
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
  src/BenchMetrics.cpp  
  src/BinMap.cpp  
  src/Enums.cpp  
  src/Net.cpp  
  src/Node.cpp  
  src/Region.cpp  
  src/Group.cpp  
  src/Params.cpp  
  src/PlaceDB.cpp  
  src/DefWriter.cpp
  src/BookshelfWriter.cpp
  src/PyPlaceDB.cpp  
  src/PybindPlaceDB.cpp  
  src/PybindPyPlaceDB.cpp  
  src/place_io.cpp  
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS}
  EXTRA_DEFINITIONS DRAWPLACE=${DRAWPLACE})

install(TARGETS 
  ${TARGET_NAME}_cpp 
  DESTINATION dreamplace/ops/${OP_NAME})
file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
