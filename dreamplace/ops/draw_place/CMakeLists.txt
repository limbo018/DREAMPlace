set(OP_NAME draw_place)

set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${LIMBO_SOURCE_DIR}
  )

set(LINK_LIBS utility_cxx
  ${LIMBO_BINARY_DIR}/limbo/parsers/gdsii/stream/libgdsparser.a
  ${LIMBO_BINARY_DIR}/limbo/thirdparty/gzstream/libgzstream.a
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
  src/draw_place.cpp 
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS}
  EXTRA_DEFINITIONS DRAWPLACE=${DRAWPLACE}
  )

install(TARGETS ${TARGET_NAME}_cpp DESTINATION dreamplace/ops/${OP_NAME})
file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
