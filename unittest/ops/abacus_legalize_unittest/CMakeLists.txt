file(GLOB SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    )
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../../dreamplace/ops")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(abacus_unittest ${SOURCES})
target_link_libraries(abacus_unittest utility_cxx)

install(
    TARGETS abacus_unittest DESTINATION unittest/ops/abacus_legalize_unittest
    )
