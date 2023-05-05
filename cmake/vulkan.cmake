if(TARGET Vulkan::Headers)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    vulkan
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG        v1.3.250
)

FetchContent_MakeAvailable(vulkan)
