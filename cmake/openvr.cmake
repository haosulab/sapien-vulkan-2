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

# HACK: disable openvr log
target_compile_options(openvr_api PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/cmake/openvr_definitions.h)
