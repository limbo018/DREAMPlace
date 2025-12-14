# Notes on packaging changes:
# - Upstream switched the demo package to .tar.gz. Version tags may vary and
#   are not guaranteed to be a specific value like v1.1.
# - Download into the build (binary) tree and extract there.

# Version and date variables for HeteroSTA releases
set(HETEROSTA_VERSION "1.1" CACHE STRING "HeteroSTA version (e.g. 1.1)")
set(HETEROSTA_VERSION_DATE "20251211" CACHE STRING "HeteroSTA version date (e.g. 20251211)")

# Construct release tag from version and date
set(HETEROSTA_RELEASE_TAG "v${HETEROSTA_VERSION}-${HETEROSTA_VERSION_DATE}" CACHE STRING "HeteroSTA release tag (e.g. v1.1-20251211) or 'latest'")

# Asset naming now encodes the version and date inside the filename, e.g. heterosta.v1.1.20251211.tar.gz.
set(HETEROSTA_ASSET_VERSION "${HETEROSTA_VERSION}.${HETEROSTA_VERSION_DATE}" CACHE STRING "Asset version used in 'heterosta.v<ver>.<date>.tar.gz'")

# Auto-generate asset name from version unless user overrides HETEROSTA_ASSET
if(NOT DEFINED HETEROSTA_ASSET_AUTOGEN)
  set(HETEROSTA_ASSET_AUTOGEN TRUE CACHE BOOL "Whether HETEROSTA_ASSET is auto-generated from HETEROSTA_ASSET_VERSION" FORCE)
endif()
if(NOT DEFINED HETEROSTA_ASSET OR HETEROSTA_ASSET_AUTOGEN)
  set(HETEROSTA_ASSET "heterosta.v${HETEROSTA_ASSET_VERSION}.tar.gz" CACHE STRING "HeteroSTA release asset filename" FORCE)
  set(HETEROSTA_ASSET_AUTOGEN TRUE CACHE BOOL "Whether HETEROSTA_ASSET is auto-generated from HETEROSTA_ASSET_VERSION" FORCE)
endif()
set(HETEROSTA_BASE_URL "https://github.com/HeteroSTA/releases/releases" CACHE STRING "Base URL for HeteroSTA release assets")

if(NOT DEFINED HETEROSTA_URL OR "${HETEROSTA_URL}" STREQUAL "")
  if(HETEROSTA_RELEASE_TAG STREQUAL "latest")
    set(HETEROSTA_URL "${HETEROSTA_BASE_URL}/latest/download/${HETEROSTA_ASSET}")
  else()
    set(HETEROSTA_URL "${HETEROSTA_BASE_URL}/download/${HETEROSTA_RELEASE_TAG}/${HETEROSTA_ASSET}")
  endif()
endif()

# Optional mirror on gitee
set(HETEROSTA_MIRROR_BASE_URL "https://gitee.com/HeteroSTA/releases/releases" CACHE STRING "Mirror base URL for HeteroSTA release assets")
if(NOT DEFINED HETEROSTA_MIRROR_URL OR "${HETEROSTA_MIRROR_URL}" STREQUAL "")
  if(HETEROSTA_RELEASE_TAG STREQUAL "latest")
    set(HETEROSTA_MIRROR_URL "${HETEROSTA_MIRROR_BASE_URL}/latest/download/${HETEROSTA_ASSET}")
  else()
    set(HETEROSTA_MIRROR_URL "${HETEROSTA_MIRROR_BASE_URL}/download/${HETEROSTA_RELEASE_TAG}/${HETEROSTA_ASSET}")
  endif()
endif()

# Place all downloaded and extracted files under the build tree.
set(HETEROSTA_SRC_DIR "${CMAKE_CURRENT_BINARY_DIR}/heterosta_src")
file(MAKE_DIRECTORY "${HETEROSTA_SRC_DIR}")

# If src is empty, try to use an existing local archive; otherwise download and extract.
file(GLOB _heterosta_src_contents "${HETEROSTA_SRC_DIR}/*")
list(LENGTH _heterosta_src_contents _heterosta_src_len)
if(_heterosta_src_len EQUAL 0)
  message(STATUS "HeteroSTA: src is empty; looking for local .tar.gz or downloading demo package ...")
  # 1) Prefer a locally present tarball in the build dir (no network needed)
  set(HETEROSTA_ARCHIVE "")
  if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${HETEROSTA_ASSET}")
    set(HETEROSTA_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/${HETEROSTA_ASSET}")
  else()
    file(GLOB _heterosta_local_archives "${CMAKE_CURRENT_BINARY_DIR}/heterosta*.tar.gz")
    list(LENGTH _heterosta_local_archives _local_len)
    if(_local_len GREATER 0)
      list(GET _heterosta_local_archives 0 HETEROSTA_ARCHIVE)
      message(STATUS "HeteroSTA: found local archive: ${HETEROSTA_ARCHIVE}; will extract it.")
    endif()
  endif()

  # 2) If no local archive, try to download from primary then mirror
  if(NOT HETEROSTA_ARCHIVE)
    set(HETEROSTA_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/${HETEROSTA_ASSET}")
    set(_download_ok FALSE)
    set(_download_urls "${HETEROSTA_URL}")
    if(DEFINED HETEROSTA_MIRROR_URL AND NOT "${HETEROSTA_MIRROR_URL}" STREQUAL "")
      list(APPEND _download_urls "${HETEROSTA_MIRROR_URL}")
    endif()

    foreach(_u IN LISTS _download_urls)
      message(STATUS "HeteroSTA: downloading '${HETEROSTA_ASSET}' from ${_u} ...")
      file(REMOVE "${HETEROSTA_ARCHIVE}")
      file(DOWNLOAD "${_u}" "${HETEROSTA_ARCHIVE}" SHOW_PROGRESS STATUS _dl_status)
      list(LENGTH _dl_status _dl_status_len)
      if(_dl_status_len GREATER 0)
        list(GET _dl_status 0 _dl_code)
      else()
        set(_dl_code 1)
      endif()
      if(_dl_code EQUAL 0)
        set(_download_ok TRUE)
        break()
      else()
        message(WARNING "HeteroSTA: download failed from ${_u} (STATUS=${_dl_status}). Trying next source if available ...")
      endif()
    endforeach()

    if(NOT _download_ok)
      file(REMOVE "${HETEROSTA_ARCHIVE}")
      message(FATAL_ERROR "HeteroSTA: failed to obtain the asset archive.\nTried downloading from: \n  - ${HETEROSTA_URL}\n  - ${HETEROSTA_MIRROR_URL}\nManual options:\n  A) Place an existing 'heterosta*.tar.gz' file into: ${CMAKE_CURRENT_BINARY_DIR}\n     (the configure step will auto-detect and extract it on next run); OR\n  B) Download the expected asset and save as: ${CMAKE_CURRENT_BINARY_DIR}/${HETEROSTA_ASSET}\n     then extract into: ${HETEROSTA_SRC_DIR}\n     e.g. 'mkdir -p \"${HETEROSTA_SRC_DIR}\" && tar -xzf \"${CMAKE_CURRENT_BINARY_DIR}/${HETEROSTA_ASSET}\" -C \"${HETEROSTA_SRC_DIR}\"'\n")
    endif()
  endif()

  # Extract .tar.gz (preferred), fallback to tar/7z/bsdtar if needed.
  set(_extract_ok FALSE)
  if(HETEROSTA_ARCHIVE MATCHES "\\.(tar\\.gz|tgz)$")
    message(STATUS "HeteroSTA: extracting with cmake -E tar ...")
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E tar xzf "${HETEROSTA_ARCHIVE}"
      WORKING_DIRECTORY "${HETEROSTA_SRC_DIR}"
      RESULT_VARIABLE _extract_rv
    )
    if(_extract_rv EQUAL 0)
      set(_extract_ok TRUE)
    else()
      find_program(HETEROSTA_TAR NAMES tar)
      if(HETEROSTA_TAR)
        message(STATUS "HeteroSTA: cmake -E tar failed; trying ${HETEROSTA_TAR} ...")
        execute_process(
          COMMAND "${HETEROSTA_TAR}" -xzf "${HETEROSTA_ARCHIVE}" -C "${HETEROSTA_SRC_DIR}"
          RESULT_VARIABLE _extract_rv
        )
        if(_extract_rv EQUAL 0)
          set(_extract_ok TRUE)
        endif()
      endif()
    endif()
  elseif(HETEROSTA_ARCHIVE MATCHES "\\.7z$")
    find_program(HETEROSTA_7Z NAMES 7z 7za 7zr)
    if(HETEROSTA_7Z)
      message(STATUS "HeteroSTA: extracting 7z with ${HETEROSTA_7Z}")
      execute_process(
        COMMAND "${HETEROSTA_7Z}" x -y -o"${HETEROSTA_SRC_DIR}" "${HETEROSTA_ARCHIVE}"
        RESULT_VARIABLE _extract_rv
      )
      if(_extract_rv EQUAL 0)
        set(_extract_ok TRUE)
      endif()
    endif()
  endif()

  if(NOT _extract_ok)
    # Last-resort: bsdtar can handle many formats.
    find_program(HETEROSTA_BSDTAR NAMES bsdtar)
    if(HETEROSTA_BSDTAR)
      message(STATUS "HeteroSTA: fallback extraction with ${HETEROSTA_BSDTAR}")
      execute_process(
        COMMAND "${HETEROSTA_BSDTAR}" -x -f "${HETEROSTA_ARCHIVE}" -C "${HETEROSTA_SRC_DIR}"
        RESULT_VARIABLE _extract_rv
      )
      if(_extract_rv EQUAL 0)
        set(_extract_ok TRUE)
      endif()
    endif()
  endif()

  if(NOT _extract_ok)
    message(FATAL_ERROR "HeteroSTA: extraction failed for ${HETEROSTA_ARCHIVE}.\nTried: 'cmake -E tar', 'tar', '7z', and 'bsdtar'.")
  endif()
endif()

# Locate package root
set(HETEROSTA_ROOT_DIR "")

# 1) include/ and lib/ directly under src
if(EXISTS "${HETEROSTA_SRC_DIR}/include" AND EXISTS "${HETEROSTA_SRC_DIR}/lib")
  set(HETEROSTA_ROOT_DIR "${HETEROSTA_SRC_DIR}")
else()
  # 2) include/ and lib/ under a single top-level child (e.g., heterosta.v1.1)
  file(GLOB _heterosta_children LIST_DIRECTORIES true "${HETEROSTA_SRC_DIR}/*")
  foreach(_child IN LISTS _heterosta_children)
    if(IS_DIRECTORY "${_child}" AND EXISTS "${_child}/include" AND EXISTS "${_child}/lib")
      set(HETEROSTA_ROOT_DIR "${_child}")
      break()
    endif()
  endforeach()
endif()

if(NOT HETEROSTA_ROOT_DIR)
  message(FATAL_ERROR "HeteroSTA: could not locate expected layout under ${HETEROSTA_SRC_DIR}.\nExpected a directory with 'include' and 'lib' (e.g., heterosta.vX.Y).")
endif()

# Determine include directory
set(HETEROSTA_INCLUDE_DIR "")
if(EXISTS "${HETEROSTA_ROOT_DIR}/include")
  set(HETEROSTA_INCLUDE_DIR "${HETEROSTA_ROOT_DIR}/include")
elseif(EXISTS "${HETEROSTA_ROOT_DIR}/heterosta.h")
  set(HETEROSTA_INCLUDE_DIR "${HETEROSTA_ROOT_DIR}")
endif()

if(NOT HETEROSTA_INCLUDE_DIR)
  message(FATAL_ERROR "HeteroSTA: include dir not found (checked ${HETEROSTA_ROOT_DIR}/include and ${HETEROSTA_ROOT_DIR})")
endif()

# Find libraries to install and pick one to link
set(HETEROSTA_LIB)

set(_candidate_lib_dir "${HETEROSTA_ROOT_DIR}/lib")
if(EXISTS "${_candidate_lib_dir}")
  file(GLOB _lib_in_root "${_candidate_lib_dir}/*.so" "${_candidate_lib_dir}/*.a")
else()
  file(GLOB _lib_in_root "${HETEROSTA_ROOT_DIR}/*.so" "${HETEROSTA_ROOT_DIR}/*.a")
endif()
list(APPEND HETEROSTA_LIB ${_lib_in_root})

if(NOT HETEROSTA_LIB)
  # recursive fallback
  file(GLOB_RECURSE _lib_any "${HETEROSTA_ROOT_DIR}/*.so" "${HETEROSTA_ROOT_DIR}/*.a")
  list(APPEND HETEROSTA_LIB ${_lib_any})
endif()

if(NOT HETEROSTA_LIB)
  message(FATAL_ERROR "HeteroSTA: no library (.so/.a) found under ${HETEROSTA_ROOT_DIR}")
endif()

set(_HETEROSTA_LINK_LIB "")


# Keep a manual override for edge cases.
set(HETEROSTA_LINK_LIB "" CACHE STRING "Explicit path to the GPU heterosta .so to link (optional)")
if(HETEROSTA_LINK_LIB)
  set(_HETEROSTA_LINK_LIB "${HETEROSTA_LINK_LIB}")
else()
  set(_HETEROSTA_LINK_LIB "")
  foreach(_lib IN LISTS HETEROSTA_LIB)
    if(_lib MATCHES "/libheterosta\\.so$")
      set(_HETEROSTA_LINK_LIB "${_lib}")
      break()
    endif()
  endforeach()
  if(NOT _HETEROSTA_LINK_LIB)
    message(FATAL_ERROR "HeteroSTA: GPU library 'libheterosta.so' not found under ${HETEROSTA_ROOT_DIR}")
  endif()
endif()

message(STATUS "HeteroSTA: root='${HETEROSTA_ROOT_DIR}'. Linking '${_HETEROSTA_LINK_LIB}', installing libs: '${HETEROSTA_LIB}' and include '${HETEROSTA_INCLUDE_DIR}'")

# Expose include dir and libraries for consumers
set(HETEROSTA_INCLUDE_DIR "${HETEROSTA_INCLUDE_DIR}" CACHE PATH "HeteroSTA include directory")
set(HETEROSTA_LIB "${HETEROSTA_LIB}" CACHE STRING "HeteroSTA libraries to install")

# Provide imported target so other parts can link with 'heterosta'
add_library(heterosta SHARED IMPORTED GLOBAL)
set_target_properties(heterosta PROPERTIES
  IMPORTED_LOCATION "${_HETEROSTA_LINK_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${HETEROSTA_INCLUDE_DIR}")
