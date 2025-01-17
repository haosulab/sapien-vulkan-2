include(FetchContent)
FetchContent_Declare(
    openvr
    GIT_REPOSITORY https://github.com/ValveSoftware/openvr.git
    GIT_TAG        v2.2.3
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)

# FetchContent_MakeAvailable(openvr)
if(NOT openvr_POPULATED)
  FetchContent_Populate(openvr)
  add_subdirectory(${openvr_SOURCE_DIR} ${openvr_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

if (MSVC)
  target_compile_definitions(openvr_api64 INTERFACE OPENVR_BUILD_STATIC)
endif()

# HACK: disable openvr log
if (NOT MSVC)
  target_compile_options(openvr_api PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/cmake/openvr_definitions.h)
endif()

# apply patch
if(APPLE)
  set(PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/patches/openvr-v2.2.3-for-apple.patch")
  execute_process(
      COMMAND git apply --check ${PATCH_FILE}
      WORKING_DIRECTORY ${openvr_SOURCE_DIR}
      RESULT_VARIABLE PATCH_RESULT
      OUTPUT_QUIET
      ERROR_QUIET
  )
  if(PATCH_RESULT EQUAL 0)
      execute_process(
          COMMAND git apply ${PATCH_FILE}
          WORKING_DIRECTORY ${openvr_SOURCE_DIR}
          RESULT_VARIABLE APPLY_RESULT
          OUTPUT_VARIABLE APPLY_OUTPUT
          ERROR_VARIABLE APPLY_ERROR
      )
      if(NOT APPLY_RESULT EQUAL 0)
          message(FATAL_ERROR "Failed to apply openvr patch:\n${APPLY_OUTPUT}\n${APPLY_ERROR}")
      else()
          message(STATUS "Openvr patch applied successfully.")
      endif()
  else()
      message(STATUS "Openvr patch already applied.")
  endif()
endif()