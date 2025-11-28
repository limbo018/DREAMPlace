# Bootstrap HeteroSTA: download and extract demo archive on first configure.
# We keep this logic local to the HeteroSTA thirdparty folder to encapsulate
# the dependency and avoid polluting the parent thirdparty CMake file.

set(HETEROSTA_URL "https://github.com/HeteroSTA/releases/releases/download/v1.0-prerelease/heterosta_demo.7z")
set(HETEROSTA_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(MAKE_DIRECTORY "${HETEROSTA_SRC_DIR}")

# If src is empty, fetch and extract.
file(GLOB _heterosta_src_contents "${HETEROSTA_SRC_DIR}/*")
list(LENGTH _heterosta_src_contents _heterosta_src_len)
if(_heterosta_src_len EQUAL 0)
  message(STATUS "HeteroSTA: src is empty; downloading demo package ...")
  set(HETEROSTA_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/heterosta_demo.7z")

  # Download the archive (no hash provided by upstream; consider adding one if available)
  file(DOWNLOAD
       "${HETEROSTA_URL}"
       "${HETEROSTA_ARCHIVE}"
       SHOW_PROGRESS
  )

  # Try to find an extractor: prefer 7z, then 7za/7zr, then bsdtar
  find_program(HETEROSTA_7Z NAMES 7z 7za 7zr)
  if(HETEROSTA_7Z)
    message(STATUS "HeteroSTA: extracting with ${HETEROSTA_7Z}")
    execute_process(
      COMMAND "${HETEROSTA_7Z}" x -y -o"${HETEROSTA_SRC_DIR}" "${HETEROSTA_ARCHIVE}"
      RESULT_VARIABLE _extract_rv
    )
    if(NOT _extract_rv EQUAL 0)
      message(FATAL_ERROR "HeteroSTA: 7z extraction failed with code ${_extract_rv}")
    endif()
  else()
    find_program(HETEROSTA_BSDTAR NAMES bsdtar)
    if(HETEROSTA_BSDTAR)
      message(STATUS "HeteroSTA: extracting with ${HETEROSTA_BSDTAR}")
      execute_process(
        COMMAND "${HETEROSTA_BSDTAR}" -x -f "${HETEROSTA_ARCHIVE}" -C "${HETEROSTA_SRC_DIR}"
        RESULT_VARIABLE _extract_rv
      )
      if(NOT _extract_rv EQUAL 0)
        message(FATAL_ERROR "HeteroSTA: bsdtar extraction failed with code ${_extract_rv}")
      endif()
    else()
      message(FATAL_ERROR "HeteroSTA: no extractor found. Please install p7zip (7z) or libarchive-tools (bsdtar).")
    endif()
  endif()
endif()

# Prefer GPU demo layout
set(HETEROSTA_GPU_DIR "${HETEROSTA_SRC_DIR}/gpu_demo")
if(NOT EXISTS "${HETEROSTA_GPU_DIR}")
  # if archive wrapped in a top-level dir, try to locate gpu_demo under it
  file(GLOB _heterosta_children LIST_DIRECTORIES true "${HETEROSTA_SRC_DIR}/*")
  foreach(_child IN LISTS _heterosta_children)
    if(IS_DIRECTORY "${_child}/gpu_demo")
      set(HETEROSTA_GPU_DIR "${_child}/gpu_demo")
      break()
    endif()
  endforeach()
endif()

if(NOT EXISTS "${HETEROSTA_GPU_DIR}")
  message(FATAL_ERROR "HeteroSTA: gpu_demo folder not found after extraction under ${HETEROSTA_SRC_DIR}")
endif()



# Determine include directory: some packages put headers directly under gpu_demo
set(HETEROSTA_INCLUDE_DIR "")
if(EXISTS "${HETEROSTA_GPU_DIR}/include")
  set(HETEROSTA_INCLUDE_DIR "${HETEROSTA_GPU_DIR}/include")
elseif(EXISTS "${HETEROSTA_GPU_DIR}/heterosta.h")
  set(HETEROSTA_INCLUDE_DIR "${HETEROSTA_GPU_DIR}")
endif()

if(NOT HETEROSTA_INCLUDE_DIR)
  message(FATAL_ERROR "HeteroSTA: include dir not found (checked ${HETEROSTA_GPU_DIR}/include and ${HETEROSTA_GPU_DIR})")
endif()

# Find libraries to install and pick one to link
set(HETEROSTA_LIB)

file(GLOB _lib_in_root "${HETEROSTA_GPU_DIR}/*.so" "${HETEROSTA_GPU_DIR}/*.a")
list(APPEND HETEROSTA_LIB ${_lib_in_root})

if(NOT HETEROSTA_LIB)
  # recursive fallback
  file(GLOB_RECURSE _lib_any "${HETEROSTA_GPU_DIR}/*.so" "${HETEROSTA_GPU_DIR}/*.a")
  list(APPEND HETEROSTA_LIB ${_lib_any})
endif()

if(NOT HETEROSTA_LIB)
  message(FATAL_ERROR "HeteroSTA: no library (.so/.a) found under ${HETEROSTA_GPU_DIR}")
endif()

set(_HETEROSTA_LINK_LIB "")
foreach(_lib IN LISTS HETEROSTA_LIB)
  if(_lib MATCHES "heterosta")
    set(_HETEROSTA_LINK_LIB "${_lib}")
    break()
  endif()
endforeach()
if(NOT _HETEROSTA_LINK_LIB)
  list(GET HETEROSTA_LIB 0 _HETEROSTA_LINK_LIB)
endif()

message(STATUS "HeteroSTA: linking '${_HETEROSTA_LINK_LIB}', installing libs: '${HETEROSTA_LIB}' and include '${HETEROSTA_INCLUDE_DIR}'")

# Expose libraries list for install step in timing_heterosta
set(HETEROSTA_LIB "${HETEROSTA_LIB}" CACHE STRING "HeteroSTA libraries to install")

# Provide imported target so other parts can link with 'heterosta'
add_library(heterosta SHARED IMPORTED GLOBAL)
set_target_properties(heterosta PROPERTIES
  IMPORTED_LOCATION "${_HETEROSTA_LINK_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${HETEROSTA_INCLUDE_DIR}")

# Copy header files into ${HETEROSTA_BINARY_DIR} path 
file(GLOB _heterosta_headers
  "${HETEROSTA_INCLUDE_DIR}/*.h" "${HETEROSTA_INCLUDE_DIR}/*.hpp" )
if(_heterosta_headers)
  file(COPY ${_heterosta_headers} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endif()
