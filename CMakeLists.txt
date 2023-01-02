cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(sapienvulkan2 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -Wall -Wnon-virtual-dtor -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g0 -Wall -Wnon-virtual-dtor -Wno-deprecated-declarations")

option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
option(SVULKAN2_PROFILE "Build with easy_profiler" OFF)
option(SVULKAN2_BUILD_TEST "Build with unit test" OFF)
option(SVULKAN2_CUDA_INTEROP "Allow CUDA to use Vulkan buffer" ON)
option(SVULKAN2_TRACK_ALLOCATION "Track allocated resources ofr debugging" OFF)

add_definitions(-DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC)

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
find_package(assimp 5.0.1 REQUIRED PACKAGE_FIND_VERSION)
find_package(Vulkan REQUIRED)
find_package(Ktx REQUIRED)
find_package(glfw3 REQUIRED)

include_directories("include")
include_directories("3rd_party/glslang")
include_directories("3rd_party/SPIRV-Cross")
include_directories("3rd_party/VulkanMemoryAllocator")
include_directories("$ENV{VULKAN_SDK}/include")
include_directories("3rd_party/stb")
link_directories("$ENV{VULKAN_SDK}/lib")
include_directories("3rd_party/imgui" "3rd_party/imgui/backends" "3rd_party/ImGuizmo")

if (${SVULKAN2_TRACK_ALLOCATION})
    add_definitions(-DTRACK_ALLOCATION)
endif()

if (${SVULKAN2_CUDA_INTEROP})
    message("-- CUDA enabled in SVulkan2 ")
    add_definitions(-DSVULKAN2_CUDA_INTEROP)
    include_directories("$ENV{CUDA_PATH}/include")
else()
    message("-- CUDA not enabled in SVulkan2 ")
endif()

if (${SVULKAN2_BUILD_TEST})
  add_subdirectory("3rd_party/googletest")
  include_directories("3rd_party/googletest/googletest/include")
endif()

if (${SVULKAN2_PROFILE})
  find_package(easy_profiler REQUIRED)
  add_definitions(-DBUILD_WITH_EASY_PROFILER)
  set(EASY_PROFILER_LIBRARY easy_profiler)
endif()

add_subdirectory("3rd_party/glslang")
add_subdirectory("3rd_party/SPIRV-Cross")

file(GLOB_RECURSE RENDER_SRC "src/*.cpp")
file(GLOB GUI_SRC "3rd_party/imgui/*.cpp"
  "3rd_party/ImGuizmo/*.cpp"
  "3rd_party/imgui/backends/imgui_impl_glfw.cpp"
  "3rd_party/imgui/backends/imgui_impl_vulkan.cpp" )

add_library(svulkan2 STATIC ${RENDER_SRC} ${GUI_SRC})

target_link_libraries(svulkan2 ${ASSIMP_LIBRARIES} glfw dl pthread Vulkan::Vulkan spdlog
    glslang SPIRV spirv-cross-cpp
    ${EASY_PROFILER_LIBRARY} ktx)

if (${SVULKAN2_CUDA_INTEROP})
    target_link_libraries(svulkan2 "$ENV{CUDA_PATH}/lib64/libcudart_static.a" rt)
endif()

set_target_properties(svulkan2 PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(spirv-cross-core PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_executable(svulkan2_main app/main.cpp)
if (MACOS)
  target_link_libraries(svulkan2_main svulkan2 "-framework Cocoa -framework IOKit")
else ()
  target_link_libraries(svulkan2_main svulkan2 stdc++fs)
endif ()

add_executable(gencube app/gencube.cpp)
if (MACOS)
    target_link_libraries(gencube svulkan2 "-framework Cocoa -framework IOKit")
else ()
    target_link_libraries(gencube svulkan2 stdc++fs)
endif ()

if (${SVULKAN2_BUILD_TEST})
    add_executable(unittest "test/shader/rt.cpp")
    # add_executable(unittest "test/shader/shader_pack.cpp")
    target_link_libraries(unittest svulkan2 gtest_main)
endif()
