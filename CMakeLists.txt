cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(sapienvulkan2 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -Wall -Wnon-virtual-dtor -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g0 -Wall -Wnon-virtual-dtor -Wno-deprecated-declarations")

option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
option(SVULKAN2_PROFILE "Build with easy_profiler" OFF)
option(SVULKAN2_BUILD_TEST "Build with unit test" OFF)
option(SVULKAN2_CUDA_INTEROP "Allow CUDA to use Vulkan buffer" ON)
option(SVULKAN2_TRACK_ALLOCATION "Track allocated resources ofr debugging" OFF)

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

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(glm)
include(assimp)
include(spdlog)
include(glfw)
include(ktx)
include(vulkan)
include(glslang)
include(spirv-cross)
include(imgui)
include(easy_profiler)


if (${SVULKAN2_TRACK_ALLOCATION})
    add_definitions(-DTRACK_ALLOCATION)
endif()

file(GLOB_RECURSE SVULKAN2_SRC "src/*.cpp")
add_library(svulkan2 SHARED ${SVULKAN2_SRC})
set_target_properties(svulkan2 PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

if (${SVULKAN2_CUDA_INTEROP})
    find_package(CUDA REQUIRED)
    message("-- CUDA enabled in SVulkan2 ")
    target_compile_definitions(svulkan2 PUBLIC SVULKAN2_CUDA_INTEROP)
    target_include_directories(svulkan2 PUBLIC "$ENV{CUDA_PATH}/include")
else()
    message("-- CUDA not enabled in SVulkan2 ")
endif()

target_compile_definitions(imgui PRIVATE VK_NO_PROTOTYPES)

target_include_directories(svulkan2 PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(svulkan2 PRIVATE "3rd_party/stb")
target_include_directories(svulkan2 PRIVATE "3rd_party/OptiX7SDK/include")
target_compile_definitions(svulkan2 PUBLIC VK_NO_PROTOTYPES)

target_link_libraries(svulkan2 PRIVATE Vulkan::Headers glm)

find_package(Threads REQUIRED)
target_link_libraries(svulkan2 PRIVATE assimp::assimp glslang SPIRV spirv-cross-cpp glfw spdlog::spdlog easy_profiler ktx imgui ${CMAKE_DL_LIBS} Threads::Threads)

if (${SVULKAN2_PROFILE})
    target_compile_definitions(svulkan2 PUBLIC SVULKAN2_PROFILE)
endif()

if (${SVULKAN2_CUDA_INTEROP})
    target_include_directories(svulkan2 PRIVATE "$ENV{CUDA_PATH}/include")
    target_link_libraries(svulkan2 PUBLIC "$ENV{CUDA_PATH}/lib64/libcudart_static.a" rt)
endif()

add_executable(svulkan2_main app/main.cpp)
target_link_libraries(svulkan2_main svulkan2 stdc++fs glfw imgui)
target_include_directories(svulkan2_main PRIVATE "3rd_party/OptiX7SDK/include")

add_executable(gencube app/gencube.cpp)
target_link_libraries(gencube svulkan2 stdc++fs)

if (${SVULKAN2_BUILD_TEST})
    add_subdirectory("3rd_party/googletest" EXCLUDE_FROM_ALL)
    add_executable(unittest "test/shader/rt.cpp")
    target_include_directories(unittest PRIVATE "3rd_party/googletest/googletest/include")
    target_link_libraries(unittest svulkan2 gtest_main)
endif()
