#pragma once
#define GLFW_INCLUDE_VULKAN
#include "svulkan2/common/log.h"
#include <iostream>
#include <vector>
#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

namespace svulkan2 {
namespace core {
class Context;
}
namespace renderer {

struct Frame {
  vk::Image mBackbuffer;
  vk::UniqueImageView mBackbufferView;

  vk::UniqueFramebuffer mImguiFramebuffer;
  vk::UniqueCommandPool mImguiCommandPool;
  vk::UniqueCommandBuffer mImguiCommandBuffer;
};

struct VulkanFrameSemaphores {
  vk::UniqueSemaphore mImageAcquiredSemaphore;
  vk::UniqueSemaphore mImguiCompleteSemaphore;
};

class GuiWindow {
  GLFWwindow *mWindow;
  vk::UniqueSurfaceKHR mSurface;

  core::Context *mContext;
  uint32_t mMinImageCount;

  uint32_t mWidth{0};
  uint32_t mHeight{0};
  vk::SurfaceFormatKHR mSurfaceFormat{};
  vk::PresentModeKHR mPresentMode{vk::PresentModeKHR::eFifo};

  uint32_t mFrameIndex{0};
  uint32_t mSemaphoreIndex{0};
  std::vector<Frame> mFrames{};
  std::vector<VulkanFrameSemaphores> mFrameSemaphores{};

  vk::UniqueDescriptorPool mDescriptorPool;

  vk::UniqueSwapchainKHR mSwapchain{};

  // imgui render pass
  vk::UniqueRenderPass mImguiRenderPass;

  // ImVec2 mMousePos{0, 0};
  // ImVec2 mMouseDelta{0, 0};
  // ImVec2 mMouseWheelDelta{0, 0};

  bool mClosed{};

public:
  inline vk::SwapchainKHR getSwapchain() const { return mSwapchain.get(); }
  inline vk::Format getBackBufferFormat() const {
    return mSurfaceFormat.format;
  }
  inline uint32_t getWidth() const { return mWidth; }
  inline uint32_t getHeight() const { return mHeight; }
  inline uint32_t getFrameIndex() const { return mFrameIndex; }

public:
  /** Acquire new frame, poll events, call ImGui NewFrame */
  void newFrame();

  /** Send current frame to the present queue. Waits for
   * renderCopmleteSemaphore, signals frameCompleteFence. */
  bool presentFrameWithImgui(vk::Semaphore renderCompleteSemaphore,
                             vk::Fence frameCompleteFence);

  /** Create ImGui Context and init ImGui Vulkan implementation. */
  void initImgui();

  inline vk::Semaphore getImageAcquiredSemaphore() {
    return mFrameSemaphores[mSemaphoreIndex].mImageAcquiredSemaphore.get();
  }

  inline vk::Image getBackbuffer() const {
    return mFrames[mFrameIndex].mBackbuffer;
  }

  GuiWindow(core::Context &context,
            std::vector<vk::Format> const &requestFormats,
            vk::ColorSpaceKHR requestColorSpace, uint32_t width,
            uint32_t height,
            std::vector<vk::PresentModeKHR> const &requestModes,
            uint32_t minImageCount);

  GuiWindow(GuiWindow const &other) = delete;
  GuiWindow &operator=(GuiWindow const &other) = delete;

  GuiWindow(GuiWindow &&other) = default;
  GuiWindow &operator=(GuiWindow &&other) = default;

  ~GuiWindow();

  bool updateSize(uint32_t w, uint32_t h);

  inline GLFWwindow *getGLFWWindow() const { return mWindow; }

  void close();
  inline bool isClosed() const { return mClosed; }

public:
  bool mFirstFrame{true};
  ImVec2 mMousePos{0, 0};
  ImVec2 mMouseDelta{0, 0};
  ImVec2 mMouseWheelDelta{0, 0};

  bool isKeyDown(std::string const &key);
  bool isKeyPressed(std::string const &key);

  bool isShiftDown();
  bool isCtrlDown();
  bool isAltDown();
  bool isSuperDown();

  // bool isKeyDown(char key);

  // bool isKeyPressed(char key);

  ImVec2 getMouseDelta();

  ImVec2 getMouseWheelDelta();

  ImVec2 getMousePosition();

  bool isMouseKeyDown(int key);

  bool isMouseKeyClicked(int key);

private:
  /** Called at initialization time  */
  void selectSurfaceFormat(std::vector<vk::Format> const &requestFormats,
                           vk::ColorSpaceKHR requestColorSpace);
  /** Called at initialization time  */
  void selectPresentMode(std::vector<vk::PresentModeKHR> const &requestModes);

  void createGlfwWindow(uint32_t width, uint32_t height);

  /** Called when the window is resized to recreate the sawpchain */
  bool recreateSwapchain(uint32_t w, uint32_t h);

  /** Called after swapchain recreation to update ImGui related resources */
  void recreateImguiResources();
};

} // namespace renderer
} // namespace svulkan2
