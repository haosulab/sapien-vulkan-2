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

find_package(assimp REQUIRED)
find_package(Vulkan REQUIRED)
find_package(Ktx REQUIRED)
find_package(glfw3 REQUIRED)
find_package(spdlog REQUIRED)

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
  add_subdirectory("3rd_party/googletest" EXCLUDE_FROM_ALL)
  include_directories("3rd_party/googletest/googletest/include")
endif()

if (${SVULKAN2_PROFILE})
  find_package(easy_profiler REQUIRED)
  add_definitions(-DBUILD_WITH_EASY_PROFILER)
  set(EASY_PROFILER_LIBRARY easy_profiler)
endif()

add_subdirectory("3rd_party/glslang" EXCLUDE_FROM_ALL)
add_subdirectory("3rd_party/SPIRV-Cross" EXCLUDE_FROM_ALL)
set_target_properties(spirv-cross-core PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

file(GLOB_RECURSE SVULKAN2_SRC
    "src/*.cpp"
    "3rd_party/imgui/imgui.cpp"
    "3rd_party/imgui/imgui_demo.cpp"
    "3rd_party/imgui/imgui_draw.cpp"
    "3rd_party/imgui/imgui_tables.cpp"
    "3rd_party/imgui/imgui_widgets.cpp"
    "3rd_party/ImGuizmo/ImGuizmo.cpp"
    "3rd_party/imgui/backends/imgui_impl_glfw.cpp"
    "3rd_party/imgui/backends/imgui_impl_vulkan.cpp")

add_library(svulkan2 STATIC ${SVULKAN2_SRC})
set_target_properties(svulkan2 PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_include_directories(svulkan2 PUBLIC "include")
target_include_directories(svulkan2 PUBLIC "3rd_party/VulkanMemoryAllocator")
target_include_directories(svulkan2 PUBLIC "3rd_party/imgui" "3rd_party/imgui/backends" "3rd_party/ImGuizmo")
target_include_directories(svulkan2 PUBLIC "3rd_party/vulkan")

target_include_directories(svulkan2 PRIVATE "3rd_party/glslang")
target_include_directories(svulkan2 PRIVATE "3rd_party/SPIRV-Cross")
target_include_directories(svulkan2 PRIVATE "3rd_party/stb")
target_include_directories(svulkan2 PRIVATE "3rd_party/OptiX7SDK/include")
target_link_libraries(svulkan2 ${ASSIMP_LIBRARIES} glfw dl pthread spdlog glslang SPIRV spirv-cross-cpp ktx ${EASY_PROFILER_LIBRARY})

if (${SVULKAN2_CUDA_INTEROP})
    target_include_directories(svulkan2 PRIVATE "$ENV{CUDA_PATH}/include")
endif()

target_compile_definitions(svulkan2 PUBLIC VK_NO_PROTOTYPES)

if (${SVULKAN2_CUDA_INTEROP})
    target_link_libraries(svulkan2 "$ENV{CUDA_PATH}/lib64/libcudart_static.a" rt)
endif()

add_executable(svulkan2_main app/main.cpp)
target_link_libraries(svulkan2_main svulkan2 stdc++fs)
target_include_directories(svulkan2_main PRIVATE "3rd_party/OptiX7SDK/include")

add_executable(gencube app/gencube.cpp)
target_link_libraries(gencube svulkan2 stdc++fs)

if (${SVULKAN2_BUILD_TEST})
    add_executable(unittest "test/shader/rt.cpp")
    target_include_directories(unittest PRIVATE "3rd_party/googletest/googletest/include")
    target_link_libraries(unittest svulkan2 gtest_main)
endif()
