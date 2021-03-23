cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(sapienvulkan2 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-o0 -g3 -Wall -Wnon-virtual-dtor -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS_RELEASE "-o3 -g0 -Wall -Wnon-virtual-dtor -Wno-deprecated-declarations")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("-- DEBUG Build")
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_DEBUG})
  add_definitions(-DVK_VALIDATION)
  add_definitions(-D_DEBUG)
else ()
  message("-- RELEASE Build")
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_RELEASE})
  add_definitions(-DNDEBUG)
endif ()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(MACOS TRUE)
  link_directories(/usr/local/lib/)
endif ()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(assimp REQUIRED)
find_package(Vulkan REQUIRED)
file(GLOB_RECURSE RENDER_SRC "src/*.cpp")

set(SVULKAN2_PROFILE FALSE CACHE BOOL "Profile with easy_profiler")
set(SVULKAN2_BUILD_TEST TRUE CACHE BOOL "Build sapien-vulkan-2 unit test")
set(SVULKAN2_CUDA_INTEROP FALSE CACHE BOOL "Allow CUDA to use Vulkan buffer")

set(BUILD_SHARED_LIBS OFF)

include_directories("include")
include_directories("3rd_party/glslang")
include_directories("3rd_party/SPIRV-Cross")
include_directories("3rd_party/VulkanMemoryAllocator")
include_directories("$ENV{VULKAN_SDK}/include")
include_directories("3rd_party/stb")

link_directories("$ENV{VULKAN_SDK}/lib")
add_subdirectory("3rd_party/glslang")
add_subdirectory("3rd_party/SPIRV-Cross")

if (${SVULKAN2_BUILD_TEST})
  add_subdirectory("3rd_party/googletest")
endif()
if (${SVULKAN2_CUDA_INTEROP})
  add_definitions(-DCUDA_INTEROP)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_TOOLKIT_INCLUDE})
endif()

if (${SVULKAN2_PROFILE})
  include_directories("3rd_party/easy_profiler")
  add_subdirectory("3rd_party/easy_profiler")
  set(EASY_PROFILER_LIBRARIES "easy_profiler")
  add_definitions(-DBUILD_WITH_EASY_PROFILER)
endif()

find_package(glfw3 REQUIRED)
include_directories("3rd_party/imgui" "3rd_party/imgui/backends")
file(GLOB GUI_SRC "3rd_party/imgui/*.cpp"
  "3rd_party/imgui/backends/imgui_impl_glfw.cpp"
  "3rd_party/imgui/backends/imgui_impl_vulkan.cpp" )

add_library(svulkan2 STATIC ${RENDER_SRC} ${GUI_SRC})

target_link_libraries(svulkan2 ${ASSIMP_LIBRARIES} glfw3 dl pthread Vulkan::Vulkan spdlog
    glslang SPIRV spirv-cross-cpp ${CUDA_LIBRARIES} ${EASY_PROFILER_LIBRARIES})

file (GLOB_RECURSE TEST_SRC "test/*.cpp")

add_executable(svulkan2_main app/main.cpp)
if (MACOS)
  target_link_libraries(svulkan2_main svulkan2 "-framework Cocoa -framework IOKit")
else ()
  target_link_libraries(svulkan2_main svulkan2 stdc++fs)
endif ()

add_executable(svulkan2_test ${TEST_SRC})
if (MACOS)
  target_link_libraries(svulkan2_test svulkan2 gtest_main "-framework Cocoa -framework IOKit")
else ()
  target_link_libraries(svulkan2_test svulkan2 gtest_main stdc++fs)
endif ()
