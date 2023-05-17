include(FetchContent)
FetchContent_Declare(
    easy_profiler
    GIT_REPOSITORY https://github.com/yse/easy_profiler.git
    GIT_TAG v2.1.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_LIST_DIR}/easy_profiler.patch.CMakeLists.txt <SOURCE_DIR>/easy_profiler_core/CMakeLists.txt  # patch for MSVC
)

set(EASY_PROFILER_NO_GUI ON CACHE BOOL "" FORCE)
FetchContent_GetProperties(easy_profiler)
if(NOT easy_profiler_POPULATED)
  FetchContent_Populate(easy_profiler)
  add_subdirectory(${easy_profiler_SOURCE_DIR} ${easy_profiler_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
set_target_properties(easy_profiler PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
