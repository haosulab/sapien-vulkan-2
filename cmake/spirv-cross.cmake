if(TARGET spirv-cross-cpp)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    spirv-cross
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Cross.git
    GIT_TAG sdk-1.3.243.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME spirv-cross)
FetchContent_GetProperties(spirv-cross)
if(NOT spirv-cross_POPULATED)
  FetchContent_Populate(spirv-cross)
  add_subdirectory(${spirv-cross_SOURCE_DIR} ${spirv-cross_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
set_target_properties(spirv-cross-core PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(spirv-cross-cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
