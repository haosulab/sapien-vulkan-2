if(TARGET assimp::assimp)
    return()
endif()

set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_MINIZIP OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
    assimp
    GIT_REPOSITORY https://github.com/assimp/assimp.git
    GIT_TAG        v5.2.5
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME assimp)
FetchContent_GetProperties(assimp)
if(NOT assimp_POPULATED)
  FetchContent_Populate(assimp)
  add_subdirectory(${assimp_SOURCE_DIR} ${assimp_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# HACK
if (zlib_SOURCE_DIR)
  target_include_directories(assimp PRIVATE ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
endif()
