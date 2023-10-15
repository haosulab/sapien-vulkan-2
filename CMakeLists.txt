cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(sapienvulkan2 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

option(SVULKAN2_SHARED "Force svulkan2 to be shared" ON)
option(SVULKAN2_PROFILE "Build with easy_profiler" OFF)
option(SVULKAN2_BUILD_TEST "Build with unit test" OFF)
option(SVULKAN2_CUDA_INTEROP "Allow CUDA to use Vulkan buffer" ON)

if(MSVC)
    add_compile_options("/W4" "$<$<CONFIG:Release>:/O2>")
else()
    add_compile_options("-Wall"
        "$<$<CONFIG:Debug>:-g3>" "$<$<CONFIG:Debug>:-O0>"
        "$<$<CONFIG:Release>:-O3>")
endif()

add_compile_definitions("$<$<CONFIG:DEBUG>:_DEBUG>")
add_compile_definitions("$<$<CONFIG:DEBUG>:VK_VALIDATION>")
add_compile_definitions("$<$<NOT:$<CONFIG:Debug>>:NDEBUG>")

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


file(GLOB_RECURSE SVULKAN2_SRC "src/*.cpp")

if(SVULKAN2_SHARED)
    add_library(svulkan2 SHARED ${SVULKAN2_SRC})
else()
    add_library(svulkan2 ${SVULKAN2_SRC})
endif()

set_target_properties(svulkan2 PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_compile_definitions(imgui PRIVATE VK_NO_PROTOTYPES)
target_compile_definitions(svulkan2 PUBLIC VK_NO_PROTOTYPES)

target_include_directories(svulkan2 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(svulkan2 PRIVATE
    "3rd_party/stb" "3rd_party/OptiX7SDK/include"
    "3rd_party/tinygltf"
)
target_link_libraries(svulkan2 PUBLIC Vulkan::Headers glm)

find_package(Threads REQUIRED)
target_link_libraries(svulkan2 PRIVATE assimp::assimp glslang SPIRV spirv-cross-cpp glfw spdlog::spdlog easy_profiler ktx imgui ${CMAKE_DL_LIBS} Threads::Threads)

if (${SVULKAN2_PROFILE})
    target_compile_definitions(svulkan2 PUBLIC SVULKAN2_PROFILE)
endif()


if (${SVULKAN2_CUDA_INTEROP})
    find_package(CUDA REQUIRED)
    target_compile_definitions(svulkan2 PUBLIC SVULKAN2_CUDA_INTEROP)
    target_include_directories(svulkan2 PUBLIC "$ENV{CUDA_PATH}/include")
    target_link_libraries(svulkan2 PUBLIC "$ENV{CUDA_PATH}/lib64/libcudart_static.a" rt)

    include(oidn)
    target_link_libraries(svulkan2 PRIVATE OpenImageDenoise)
endif()

# TODO: add option to control the following variables
# this library exports the storage api
target_compile_definitions(svulkan2 PRIVATE VULKAN_HPP_STORAGE_SHARED_EXPORT)
# other libraries use the exported storage api
target_compile_definitions(svulkan2 PUBLIC VULKAN_HPP_STORAGE_SHARED)

install(TARGETS svulkan2 Vulkan-Headers glm OpenImageDenoise OpenImageDenoise_core OpenImageDenoise_common
    EXPORT svulkan2Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(EXPORT svulkan2Targets
    FILE svulkan2.cmake
    NAMESPACE svulkan2::
    DESTINATION lib/cmake/svulkan2
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/svulkan2 DESTINATION "include")
install(DIRECTORY "${vulkan_SOURCE_DIR}/include/vulkan" DESTINATION "include")
install(DIRECTORY "${vulkan_SOURCE_DIR}/include/vk_video" DESTINATION "include")
install(DIRECTORY "${glm_SOURCE_DIR}/glm" DESTINATION "include")
# install(IMPORTED_RUNTIME_ARTIFACTS OpenImageDenoise_device_cuda)


add_executable(svulkan2_main app/main.cpp)
target_link_libraries(svulkan2_main svulkan2 glfw imgui)
if (UNIX)
    target_link_libraries(svulkan2_main stdc++fs)
endif()

add_executable(gencube app/gencube.cpp)
target_link_libraries(gencube svulkan2)

# if (${SVULKAN2_BUILD_TEST})
#     add_subdirectory("3rd_party/googletest" EXCLUDE_FROM_ALL)
#     add_executable(unittest "test/shader/rt.cpp")
#     target_include_directories(unittest PRIVATE "3rd_party/googletest/googletest/include")
#     target_link_libraries(unittest svulkan2 gtest_main)

#     add_executable(svulkan2_test app/test.cpp)
#     target_link_libraries(svulkan2_test svulkan2 glfw imgui)
# endif()
