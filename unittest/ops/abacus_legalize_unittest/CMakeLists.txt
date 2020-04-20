get_filename_component(UTILITY_LIBRARY_DIRS ${CMAKE_CURRENT_BINARY_DIR}/../../../dreamplace/ops/utility ABSOLUTE)

file(GLOB SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    )
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../../dreamplace/ops")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set (CMAKE_CXX_STANDARD 11)

add_executable(abacus_unittest ${SOURCES})
target_link_libraries(abacus_unittest ${UTILITY_LIBRARY_DIRS}/libutility.a)

install(
    TARGETS abacus_unittest DESTINATION unittest/ops/abacus_legalize_unittest
    )
