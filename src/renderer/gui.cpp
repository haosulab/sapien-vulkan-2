#include "svulkan2/renderer/gui.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace renderer {

static void checkVKResult(VkResult err) {
  if (err != 0) {
    log::error("Vulkan result check failed.");
  }
}

static vk::UniqueRenderPass createImguiRenderPass(vk::Device device,
                                                  vk::Format format) {

  vk::AttachmentDescription attachment{{},
                                       format,
                                       vk::SampleCountFlagBits::e1,
                                       vk::AttachmentLoadOp::eLoad,
                                       vk::AttachmentStoreOp::eStore,
                                       vk::AttachmentLoadOp::eDontCare,
                                       vk::AttachmentStoreOp::eDontCare,
                                       vk::ImageLayout::eColorAttachmentOptimal,
                                       vk::ImageLayout::ePresentSrcKHR};
  vk::AttachmentReference colorAttachment{
      0, vk::ImageLayout::eColorAttachmentOptimal};

  vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, 0,
                                 nullptr, 1, &colorAttachment);
  vk::SubpassDependency dependency{
      VK_SUBPASS_EXTERNAL,
      0,
      vk::PipelineStageFlagBits::eAllCommands,
      vk::PipelineStageFlagBits::eColorAttachmentOutput,
      vk::AccessFlagBits::eMemoryWrite,
      vk::AccessFlagBits::eColorAttachmentWrite};

  vk::RenderPassCreateInfo info({}, 1, &attachment, 1, &subpass, 1,
                                &dependency);
  return device.createRenderPassUnique(info);
}

GuiWindow::GuiWindow(core::Context &context,
                     std::vector<vk::Format> const &requestFormats,
                     vk::ColorSpaceKHR requestColorSpace, uint32_t width,
                     uint32_t height,
                     std::vector<vk::PresentModeKHR> const &requestModes,
                     uint32_t minImageCount)
    : mContext(&context), mMinImageCount(minImageCount) {
  createGlfwWindow(width, height);
  selectSurfaceFormat(requestFormats, requestColorSpace);
  selectPresentMode(requestModes);
}

void GuiWindow::newFrame() {
  glfwPollEvents();
  mSemaphoreIndex = (mSemaphoreIndex + 1) % mFrameSemaphores.size();
  auto result = mContext->getDevice().acquireNextImageKHR(
      mSwapchain.get(), UINT64_MAX,
      mFrameSemaphores[mSemaphoreIndex].mImageAcquiredSemaphore.get(), {});
  if (result.result != vk::Result::eSuccess) {
    log::error("Acquire image failed");
  }
  mFrameIndex = result.value;

  auto mousePos = ImGui::GetMousePos();
  static bool firstFrame = true;
  if (firstFrame) {
    firstFrame = false;
  } else {
    mMouseDelta = {mousePos.x - mMousePos.x, mousePos.y - mMousePos.y};
  }
  mMousePos = mousePos;

  if (ImGui::GetIO().WantCaptureMouse) {
    mMouseWheelDelta = {0, 0};
  } else {
    mMouseWheelDelta = {ImGui::GetIO().MouseWheel, ImGui::GetIO().MouseWheelH};
  }

  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
}

bool GuiWindow::presentFrameWithImgui(vk::Semaphore renderCompleteSemaphore,
                                      vk::Fence frameCompleteFence) {
  vk::ClearValue clearValue{};
  mContext->getDevice().resetCommandPool(
      mFrames[mFrameIndex].mImguiCommandPool.get(), {});
  mFrames[mFrameIndex].mImguiCommandBuffer->begin(
      {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  vk::RenderPassBeginInfo info(mImguiRenderPass.get(),
                               mFrames[mFrameIndex].mImguiFramebuffer.get(),
                               {{0, 0}, {mWidth, mHeight}}, 1, &clearValue);
  mFrames[mFrameIndex].mImguiCommandBuffer->beginRenderPass(
      info, vk::SubpassContents::eInline);

  ImGui_ImplVulkan_RenderDrawData(
      ImGui::GetDrawData(), mFrames[mFrameIndex].mImguiCommandBuffer.get());
  mFrames[mFrameIndex].mImguiCommandBuffer->endRenderPass();
  mFrames[mFrameIndex].mImguiCommandBuffer->end();

  vk::PipelineStageFlags waitStage =
      vk::PipelineStageFlagBits::eColorAttachmentOutput;
  vk::SubmitInfo submitInfo{
      1,
      &renderCompleteSemaphore,
      &waitStage,
      1,
      &mFrames[mFrameIndex].mImguiCommandBuffer.get(),
      1,
      &mFrameSemaphores[mSemaphoreIndex].mImguiCompleteSemaphore.get()};
  mContext->getQueue().submit(submitInfo, frameCompleteFence);

  // the present queue should be the same as graphics queue
  return mContext->getQueue().presentKHR(
             {1,
              &mFrameSemaphores[mSemaphoreIndex].mImguiCompleteSemaphore.get(),
              1, &mSwapchain.get(), &mFrameIndex}) == vk::Result::eSuccess;
}

void GuiWindow::initImgui() {
  auto device = mContext->getDevice();

  // create a pool for allocating descriptor sets
  vk::DescriptorPoolSize pool_sizes[] = {
      {vk::DescriptorType::eSampler, 1000},
      {vk::DescriptorType::eCombinedImageSampler, 1000},
      {vk::DescriptorType::eSampledImage, 1000},
      {vk::DescriptorType::eStorageImage, 1000},
      {vk::DescriptorType::eUniformTexelBuffer, 1000},
      {vk::DescriptorType::eStorageTexelBuffer, 1000},
      {vk::DescriptorType::eUniformBuffer, 1000},
      {vk::DescriptorType::eStorageBuffer, 1000},
      {vk::DescriptorType::eUniformBufferDynamic, 1000},
      {vk::DescriptorType::eStorageBufferDynamic, 1000},
      {vk::DescriptorType::eInputAttachment, 1000}};
  auto info = vk::DescriptorPoolCreateInfo(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1000 * 11, 11,
      pool_sizes);
  mDescriptorPool = device.createDescriptorPoolUnique(info);

  mImguiRenderPass = createImguiRenderPass(device, mSurfaceFormat.format);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForVulkan(mWindow, true);
  ImGui_ImplVulkan_InitInfo initInfo = {};
  initInfo.Instance = mContext->getInstance();
  initInfo.PhysicalDevice = mContext->getPhysicalDevice();
  initInfo.Device = device;
  initInfo.QueueFamily = mContext->getGraphicsQueueFamilyIndex();
  initInfo.Queue = mContext->getQueue();

  initInfo.PipelineCache = {};
  initInfo.DescriptorPool = mDescriptorPool.get();
  initInfo.Allocator = {};
  initInfo.MinImageCount = mMinImageCount;
  initInfo.ImageCount = mMinImageCount;
  initInfo.CheckVkResultFn = checkVKResult;
  ImGui_ImplVulkan_Init(&initInfo, mImguiRenderPass.get());

  auto commandBuffer = mContext->createCommandBuffer();
  commandBuffer->begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  ImGui_ImplVulkan_CreateFontsTexture(commandBuffer.get());
  commandBuffer->end();
  mContext->submitCommandBufferAndWait(commandBuffer.get());
  log::info("Imgui initialized");
  updateSize(mWidth, mHeight);
}

void GuiWindow::createGlfwWindow(uint32_t width, uint32_t height) {
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  mWindow = glfwCreateWindow(width, height, "vulkan", nullptr, nullptr);

  VkSurfaceKHR tmpSurface;

  auto result = glfwCreateWindowSurface(mContext->getInstance(), mWindow,
                                        nullptr, &tmpSurface);
  if (result != VK_SUCCESS) {
    throw std::runtime_error(
        "create window failed: glfwCreateWindowSurface failed");
  }
  mSurface = vk::UniqueSurfaceKHR(tmpSurface, mContext->getInstance());

  if (!mContext->getPhysicalDevice().getSurfaceSupportKHR(
          mContext->getGraphicsQueueFamilyIndex(), mSurface.get())) {
    throw std::runtime_error("create window failed: graphics device does not "
                             "have present capability");
  }
}

void GuiWindow::selectSurfaceFormat(
    std::vector<vk::Format> const &requestFormats,
    vk::ColorSpaceKHR requestColorSpace) {
  assert(requestFormats.size() > 0);

  auto avail_formats =
      mContext->getPhysicalDevice().getSurfaceFormatsKHR(mSurface.get());
  if (avail_formats.size() == 0) {
    throw std::runtime_error("No surface format is available");
  }

  if (avail_formats.size() == 1) {
    if (avail_formats[0].format == vk::Format::eUndefined) {
      vk::SurfaceFormatKHR ret;
      ret.format = requestFormats[0];
      ret.colorSpace = requestColorSpace;
      mSurfaceFormat = ret;
      return;
    }
    // No point in searching another format
    mSurfaceFormat = avail_formats[0];
    return;
  }

  // Request several formats, the first found will be used
  for (uint32_t request_i = 0; request_i < requestFormats.size(); request_i++) {
    for (uint32_t avail_i = 0; avail_i < avail_formats.size(); avail_i++) {
      if (avail_formats[avail_i].format == requestFormats[request_i] &&
          avail_formats[avail_i].colorSpace == requestColorSpace) {
        mSurfaceFormat = avail_formats[avail_i];
        return;
      }
    }
  }

  log::warn("SelectSurfaceFormat: None of the requested surface formats is "
            "available");
  // If none of the requested image formats could be found, use the first
  // available
  mSurfaceFormat = avail_formats[0];
}

void GuiWindow::selectPresentMode(
    std::vector<vk::PresentModeKHR> const &requestModes) {
  assert(requestModes.size() > 0);

  // Request a certain mode and confirm that it is available. If not use
  // VK_PRESENT_MODE_FIFO_KHR which is mandatory
  auto avail_modes =
      mContext->getPhysicalDevice().getSurfacePresentModesKHR(mSurface.get());

  for (uint32_t request_i = 0; request_i < requestModes.size(); request_i++) {
    for (uint32_t avail_i = 0; avail_i < avail_modes.size(); avail_i++) {
      if (requestModes[request_i] == avail_modes[avail_i]) {
        mPresentMode = requestModes[request_i];
        return;
      }
    }
  }

  mPresentMode = vk::PresentModeKHR::eFifo; // always available
}

void GuiWindow::updateSize(uint32_t w, uint32_t h) {
  recreateSwapchain(w, h);
  recreateImguiResources();
  mContext->getDevice().waitIdle();
}

void GuiWindow::recreateSwapchain(uint32_t w, uint32_t h) {
  if (mMinImageCount == 0) {
    throw std::runtime_error("Invalid min image count specified");
  }

  vk::SwapchainCreateInfoKHR info({}, mSurface.get(), mMinImageCount,
                                  mSurfaceFormat.format,
                                  mSurfaceFormat.colorSpace, {w, h}, 1,
                                  vk::ImageUsageFlagBits::eColorAttachment |
                                      vk::ImageUsageFlagBits::eTransferDst,
                                  vk::SharingMode::eExclusive, 0, nullptr,
                                  vk::SurfaceTransformFlagBitsKHR::eIdentity,
                                  vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                  mPresentMode, VK_TRUE, mSwapchain.get());
  auto device = mContext->getDevice();
  auto cap =
      mContext->getPhysicalDevice().getSurfaceCapabilitiesKHR(mSurface.get());
  if (info.minImageCount < cap.minImageCount) {
    info.minImageCount = cap.minImageCount;
  } else if (cap.maxImageCount != 0 && info.minImageCount > cap.maxImageCount) {
    info.minImageCount = cap.maxImageCount;
  }
  if (cap.currentExtent.width != 0xffffffff) {
    mWidth = info.imageExtent.width = cap.currentExtent.width;
    mHeight = info.imageExtent.height = cap.currentExtent.height;
  } else {
    mWidth = w;
    mWidth = h;
  }
  mSwapchain = device.createSwapchainKHRUnique(info);
  auto images = device.getSwapchainImagesKHR(mSwapchain.get());

  assert(images.size() >= mMinImageCount);
  assert(images.size() < 16);

  mFrames.resize(images.size());
  mFrameSemaphores.resize(images.size());
  for (uint32_t i = 0; i < images.size(); ++i) {
    mFrameSemaphores[i].mImageAcquiredSemaphore =
        device.createSemaphoreUnique({});
    mFrames[i].mBackbuffer = images[i];
    vk::ImageViewCreateInfo info{
        {},
        mFrames[i].mBackbuffer,
        vk::ImageViewType::e2D,
        mSurfaceFormat.format,
        {vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
         vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA},
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
    mFrames[i].mBackbufferView = device.createImageViewUnique(info);
  }
}

void GuiWindow::recreateImguiResources() {
  auto device = mContext->getDevice();
  for (uint32_t i = 0; i < mFrames.size(); ++i) {
    mFrames[i].mImguiCommandBuffer.reset();
    mFrames[i].mImguiCommandPool.reset();
  }
  for (uint32_t i = 0; i < mFrames.size(); ++i) {
    mFrames[i].mImguiCommandPool = device.createCommandPoolUnique(
        {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
         mContext->getGraphicsQueueFamilyIndex()});
    mFrames[i].mImguiCommandBuffer = std::move(
        device
            .allocateCommandBuffersUnique({mFrames[i].mImguiCommandPool.get(),
                                           vk::CommandBufferLevel::ePrimary, 1})
            .front());

    vk::FramebufferCreateInfo info({}, mImguiRenderPass.get(), 1,
                                   &mFrames[i].mBackbufferView.get(), mWidth,
                                   mHeight, 1);
    mFrames[i].mImguiFramebuffer = device.createFramebufferUnique(info);
  }
  for (uint32_t i = 0; i < mFrameSemaphores.size(); ++i) {
    mFrameSemaphores[i].mImguiCompleteSemaphore =
        device.createSemaphoreUnique({});
  }
}

void GuiWindow::close() {
  if (!mClosed) {
    mClosed = true;
    glfwSetWindowShouldClose(mWindow, true);

    mContext->getDevice().waitIdle();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    mImguiRenderPass.reset();
    mFrameSemaphores.clear();
    mFrames.clear();
    mSwapchain.reset();
    mSurface.reset();
  }
}

GuiWindow::~GuiWindow() {
  close();
  glfwDestroyWindow(mWindow);
}

static int findKeyCode(std::string const &key) {
  static std::unordered_map<std::string, int> keyMap = {
      {"a", ImGui::GetKeyIndex(ImGuiKey_A)},
      {"b", ImGui::GetKeyIndex(ImGuiKey_A) + 1},
      {"c", ImGui::GetKeyIndex(ImGuiKey_A) + 2},
      {"d", ImGui::GetKeyIndex(ImGuiKey_A) + 3},
      {"e", ImGui::GetKeyIndex(ImGuiKey_A) + 4},
      {"f", ImGui::GetKeyIndex(ImGuiKey_A) + 5},
      {"g", ImGui::GetKeyIndex(ImGuiKey_A) + 6},
      {"h", ImGui::GetKeyIndex(ImGuiKey_A) + 7},
      {"i", ImGui::GetKeyIndex(ImGuiKey_A) + 8},
      {"j", ImGui::GetKeyIndex(ImGuiKey_A) + 9},
      {"k", ImGui::GetKeyIndex(ImGuiKey_A) + 10},
      {"l", ImGui::GetKeyIndex(ImGuiKey_A) + 11},
      {"m", ImGui::GetKeyIndex(ImGuiKey_A) + 12},
      {"n", ImGui::GetKeyIndex(ImGuiKey_A) + 13},
      {"o", ImGui::GetKeyIndex(ImGuiKey_A) + 14},
      {"p", ImGui::GetKeyIndex(ImGuiKey_A) + 15},
      {"q", ImGui::GetKeyIndex(ImGuiKey_A) + 16},
      {"r", ImGui::GetKeyIndex(ImGuiKey_A) + 17},
      {"s", ImGui::GetKeyIndex(ImGuiKey_A) + 18},
      {"t", ImGui::GetKeyIndex(ImGuiKey_A) + 19},
      {"u", ImGui::GetKeyIndex(ImGuiKey_A) + 20},
      {"v", ImGui::GetKeyIndex(ImGuiKey_A) + 21},
      {"w", ImGui::GetKeyIndex(ImGuiKey_A) + 22},
      {"x", ImGui::GetKeyIndex(ImGuiKey_A) + 23},
      {"y", ImGui::GetKeyIndex(ImGuiKey_A) + 24},
      {"z", ImGui::GetKeyIndex(ImGuiKey_A) + 25},
      {" ", ImGui::GetKeyIndex(ImGuiKey_Space)},
      {"space", ImGui::GetKeyIndex(ImGuiKey_Space)},
      {"tab", ImGui::GetKeyIndex(ImGuiKey_Tab)},
      {"enter", ImGui::GetKeyIndex(ImGuiKey_Enter)},
      {"insert", ImGui::GetKeyIndex(ImGuiKey_Insert)},
      {"home", ImGui::GetKeyIndex(ImGuiKey_Home)},
      {"delete", ImGui::GetKeyIndex(ImGuiKey_Delete)},
      {"end", ImGui::GetKeyIndex(ImGuiKey_End)},
      {"pageup", ImGui::GetKeyIndex(ImGuiKey_PageUp)},
      {"pagedown", ImGui::GetKeyIndex(ImGuiKey_PageDown)},
      {"up", ImGui::GetKeyIndex(ImGuiKey_UpArrow)},
      {"down", ImGui::GetKeyIndex(ImGuiKey_DownArrow)},
      {"left", ImGui::GetKeyIndex(ImGuiKey_LeftArrow)},
      {"right", ImGui::GetKeyIndex(ImGuiKey_RightArrow)}};
  if (keyMap.find(key) == keyMap.end()) {
    throw std::runtime_error("unknown key " + key);
  }
  return keyMap.at(key);
}

bool GuiWindow::isKeyDown(std::string const &key) {
  int code = findKeyCode(key);
  if (ImGui::GetIO().WantTextInput || ImGui::GetIO().WantCaptureKeyboard) {
    return false;
  }
  return ImGui::IsKeyDown(code);
}

bool GuiWindow::isKeyPressed(std::string const &key) {
  int code = findKeyCode(key);
  if (ImGui::GetIO().WantTextInput || ImGui::GetIO().WantCaptureKeyboard) {
    return false;
  }
  return ImGui::IsKeyPressed(code);
}

bool GuiWindow::isShiftDown() { return ImGui::GetIO().KeyShift; }
bool GuiWindow::isCtrlDown() { return ImGui::GetIO().KeyCtrl; }
bool GuiWindow::isAltDown() { return ImGui::GetIO().KeyAlt; }
bool GuiWindow::isSuperDown() { return ImGui::GetIO().KeySuper; }

ImVec2 GuiWindow::getMouseDelta() {
  mMouseDelta.x = std::clamp(mMouseDelta.x, -100.f, 100.f);
  mMouseDelta.y = std::clamp(mMouseDelta.y, -100.f, 100.f);
  return mMouseDelta;
}

ImVec2 GuiWindow::getMouseWheelDelta() { return mMouseWheelDelta; }

ImVec2 GuiWindow::getMousePosition() { return mMousePos; }

bool GuiWindow::isMouseKeyDown(int key) {
  return !ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseDown(key);
}

bool GuiWindow::isMouseKeyClicked(int key) {
  return !ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseClicked(key);
}

} // namespace renderer
} // namespace svulkan2
