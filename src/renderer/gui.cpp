#include "svulkan2/renderer/gui.h"
#include "svulkan2/common/fonts/roboto.hpp"
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

static void windowCallback(GLFWwindow *window, int count, const char **paths) {
  static_cast<GuiWindow *>(glfwGetWindowUserPointer(window))
      ->dropCallback(count, paths);
}

GuiWindow::GuiWindow(std::vector<vk::Format> const &requestFormats,
                     vk::ColorSpaceKHR requestColorSpace, uint32_t width,
                     uint32_t height,
                     std::vector<vk::PresentModeKHR> const &requestModes,
                     uint32_t minImageCount)
    : mMinImageCount(minImageCount) {
  mContext = core::Context::Get();
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
  if (result.result == vk::Result::eSuboptimalKHR) {
    throw vk::OutOfDateKHRError("Suboptimal");
  } else if (result.result != vk::Result::eSuccess) {
    throw std::runtime_error("Acquire image failed");
  }
  mFrameIndex = result.value;

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
  mContext->getQueue().submit(
      mFrames[mFrameIndex].mImguiCommandBuffer.get(), renderCompleteSemaphore,
      waitStage,
      mFrameSemaphores[mSemaphoreIndex].mImguiCompleteSemaphore.get(),
      frameCompleteFence);
  return mContext->getQueue().present(
             mFrameSemaphores[mSemaphoreIndex].mImguiCompleteSemaphore.get(),
             mSwapchain.get(), mFrameIndex) == vk::Result::eSuccess;
}

static void applyStyle() {
  ImGuiStyle &style = ImGui::GetStyle();
  ImVec4 *colors = style.Colors;
  colors[ImGuiCol_Text] = ImVec4(1.000f, 1.000f, 1.000f, 1.000f);
  colors[ImGuiCol_TextDisabled] = ImVec4(0.500f, 0.500f, 0.500f, 1.000f);
  colors[ImGuiCol_WindowBg] = ImVec4(0.180f, 0.180f, 0.180f, 0.900f);
  colors[ImGuiCol_ChildBg] = ImVec4(0.280f, 0.280f, 0.280f, 0.000f);
  colors[ImGuiCol_PopupBg] = ImVec4(0.313f, 0.313f, 0.313f, 1.000f);
  colors[ImGuiCol_Border] = ImVec4(0.266f, 0.266f, 0.266f, 1.000f);
  colors[ImGuiCol_BorderShadow] = ImVec4(0.000f, 0.000f, 0.000f, 0.000f);
  colors[ImGuiCol_FrameBg] = ImVec4(0.160f, 0.160f, 0.160f, 1.000f);
  colors[ImGuiCol_FrameBgHovered] = ImVec4(0.200f, 0.200f, 0.200f, 1.000f);
  colors[ImGuiCol_FrameBgActive] = ImVec4(0.280f, 0.280f, 0.280f, 1.000f);
  colors[ImGuiCol_TitleBg] = ImVec4(0.148f, 0.148f, 0.148f, 1.000f);
  colors[ImGuiCol_TitleBgActive] = ImVec4(0.148f, 0.148f, 0.148f, 1.000f);
  colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.148f, 0.148f, 0.148f, 1.000f);
  colors[ImGuiCol_MenuBarBg] = ImVec4(0.195f, 0.195f, 0.195f, 1.000f);
  colors[ImGuiCol_ScrollbarBg] = ImVec4(0.160f, 0.160f, 0.160f, 1.000f);
  colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.277f, 0.277f, 0.277f, 1.000f);
  colors[ImGuiCol_ScrollbarGrabHovered] =
      ImVec4(0.300f, 0.300f, 0.300f, 1.000f);
  colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
  colors[ImGuiCol_CheckMark] = ImVec4(1.000f, 1.000f, 1.000f, 1.000f);
  colors[ImGuiCol_SliderGrab] = ImVec4(0.391f, 0.391f, 0.391f, 1.000f);
  colors[ImGuiCol_SliderGrabActive] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
  colors[ImGuiCol_Button] = ImVec4(1.000f, 1.000f, 1.000f, 0.100f);
  colors[ImGuiCol_ButtonHovered] = ImVec4(1.000f, 1.000f, 1.000f, 0.156f);
  colors[ImGuiCol_ButtonActive] = ImVec4(1.000f, 1.000f, 1.000f, 0.391f);
  colors[ImGuiCol_Header] = ImVec4(0.313f, 0.313f, 0.313f, 0.800f);
  colors[ImGuiCol_HeaderHovered] = ImVec4(0.469f, 0.469f, 0.469f, 1.000f);
  colors[ImGuiCol_HeaderActive] = ImVec4(0.469f, 0.469f, 0.469f, 1.000f);
  colors[ImGuiCol_Separator] = colors[ImGuiCol_Border];
  colors[ImGuiCol_SeparatorHovered] = ImVec4(0.391f, 0.391f, 0.391f, 1.000f);
  colors[ImGuiCol_SeparatorActive] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
  colors[ImGuiCol_ResizeGrip] = ImVec4(1.000f, 1.000f, 1.000f, 0.250f);
  colors[ImGuiCol_ResizeGripHovered] = ImVec4(1.000f, 1.000f, 1.000f, 0.670f);
  colors[ImGuiCol_ResizeGripActive] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
  colors[ImGuiCol_Tab] = ImVec4(0.098f, 0.098f, 0.098f, 1.000f);
  colors[ImGuiCol_TabHovered] = ImVec4(0.352f, 0.352f, 0.352f, 1.000f);
  colors[ImGuiCol_TabActive] = ImVec4(0.195f, 0.195f, 0.195f, 1.000f);
  colors[ImGuiCol_TabUnfocused] = ImVec4(0.098f, 0.098f, 0.098f, 1.000f);
  colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.195f, 0.195f, 0.195f, 1.000f);
  colors[ImGuiCol_DockingPreview] = ImVec4(1.000f, 0.391f, 0.000f, 0.781f);
  colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.180f, 0.180f, 0.180f, 1.000f);
  colors[ImGuiCol_PlotLines] = ImVec4(0.469f, 0.469f, 0.469f, 1.000f);
  colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
  colors[ImGuiCol_PlotHistogram] = ImVec4(0.586f, 0.586f, 0.586f, 1.000f);
  colors[ImGuiCol_PlotHistogramHovered] =
      ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
  colors[ImGuiCol_TextSelectedBg] = ImVec4(1.000f, 1.000f, 1.000f, 0.156f);
  colors[ImGuiCol_DragDropTarget] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
  colors[ImGuiCol_NavHighlight] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
  colors[ImGuiCol_NavWindowingHighlight] =
      ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
  colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.000f, 0.000f, 0.000f, 0.586f);
  colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.000f, 0.000f, 0.000f, 0.586f);

  style.ChildRounding = 4.0f;
  style.FrameBorderSize = 1.0f;
  style.FrameRounding = 2.0f;
  style.GrabMinSize = 7.0f;
  style.PopupRounding = 2.0f;
  style.ScrollbarRounding = 12.0f;
  style.ScrollbarSize = 13.0f;
  style.TabBorderSize = 1.0f;
  style.TabRounding = 0.0f;
  style.WindowRounding = 4.0f;
}

void GuiWindow::initImgui() {
  auto device = mContext->getDevice();

  vk::Instance instance = mContext->getInstance();
  ImGui_ImplVulkan_LoadFunctions(
      [](const char *name, void *instance) {
        return reinterpret_cast<vk::Instance *>(instance)->getProcAddr(name);
      },
      &instance);

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

  applyStyle();
  auto &io = ImGui::GetIO();

  // TODO: determine the current monitor and set scale
  {
    int monitorCount = 0;
    auto monitors = glfwGetMonitors(&monitorCount);
    for (int i = 0; i < monitorCount; ++i) {
      float xscale = 0.f;
      float yscale = 0.f;
      glfwGetMonitorContentScale(monitors[i], &xscale, &yscale);
      log::info("Monitor {} scale: {}", i, xscale);
      assert(xscale == yscale); // this should always be true
      mContentScale = std::max(yscale, std::max(xscale, mContentScale));
    }
    if (mContentScale < 0.1f) {
      mContentScale = 1.f;
    }
    log::info("Largest monitor DPI scale: {}", mContentScale);

    // HACK: do not scale content twice
    static bool __called = false;
    float fontScale = 1.f;
    if (!__called) {
      ImGui::GetStyle().ScaleAllSizes(mContentScale);
      fontScale *= mContentScale;
      __called = true;
    }

    auto font = io.Fonts->AddFontFromMemoryCompressedBase85TTF(
        roboto_compressed_data_base85, std::round(17.f * fontScale));
    if (font != nullptr) {
      io.FontDefault = font;
    }
  }

  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  ImGui_ImplGlfw_InitForVulkan(mWindow, true);
  ImGui_ImplVulkan_InitInfo initInfo = {};
  initInfo.Instance = mContext->getInstance();
  initInfo.PhysicalDevice = mContext->getPhysicalDevice();
  initInfo.Device = device;
  initInfo.QueueFamily = mContext->getGraphicsQueueFamilyIndex();
  initInfo.Queue = mContext->getQueue().getVulkanQueue();

  initInfo.PipelineCache = {};
  initInfo.DescriptorPool = mDescriptorPool.get();
  initInfo.Allocator = {};
  initInfo.MinImageCount = mMinImageCount;
  initInfo.ImageCount = mMinImageCount;
  initInfo.CheckVkResultFn = checkVKResult;
  ImGui_ImplVulkan_Init(&initInfo, mImguiRenderPass.get());

  auto pool = mContext->createCommandPool();
  auto commandBuffer = pool->allocateCommandBuffer();
  commandBuffer->begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  ImGui_ImplVulkan_CreateFontsTexture(commandBuffer.get());
  commandBuffer->end();
  mContext->getQueue().submitAndWait(commandBuffer.get());
  log::info("Imgui initialized");
  updateSize(mWidth, mHeight);
}

void GuiWindow::createGlfwWindow(uint32_t width, uint32_t height) {
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHintString(GLFW_X11_CLASS_NAME, "sapien");
  glfwWindowHintString(GLFW_X11_INSTANCE_NAME, "SAPIEN");
  mWindow = glfwCreateWindow(width, height, "SAPIEN", nullptr, nullptr);

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

  glfwSetWindowUserPointer(mWindow, this);
  glfwSetDropCallback(mWindow, windowCallback);
}

void GuiWindow::dropCallback(int count, const char **paths) {
  std::vector<std::string> v;
  for (int i = 0; i < count; ++i) {
    v.push_back(paths[i]);
  }
  if (mDropCallback) {
    mDropCallback(v);
  }
}

void GuiWindow::setDropCallback(
    std::function<void(std::vector<std::string>)> callback) {
  mDropCallback = callback;
}
void GuiWindow::unsetDropCallback() { mDropCallback = {}; }

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

bool GuiWindow::updateSize(uint32_t w, uint32_t h) {
  if (!recreateSwapchain(w, h)) {
    return false;
  }
  recreateImguiResources();
  mContext->getDevice().waitIdle();
  return true;
}

bool GuiWindow::recreateSwapchain(uint32_t w, uint32_t h) {
  if (mMinImageCount == 0) {
    throw std::runtime_error("Invalid min image count specified");
  }

  auto device = mContext->getDevice();
  auto cap =
      mContext->getPhysicalDevice().getSurfaceCapabilitiesKHR(mSurface.get());
  if (cap.minImageExtent.width > w || cap.maxImageExtent.width < w ||
      cap.minImageExtent.height > h || cap.maxImageExtent.height < h) {
    log::info("swapchain create ignored: requested size ({}, {}); available "
              "{}-{}, {}-{}",
              w, h, cap.minImageExtent.width, cap.maxImageExtent.width,
              cap.minImageExtent.height, cap.maxImageExtent.height);
    return false;
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
    mHeight = h;
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
  return true;
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
    ImGui::DestroyContext();

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
      {"esc", ImGui::GetKeyIndex(ImGuiKey_Escape)},
      {"escape", ImGui::GetKeyIndex(ImGuiKey_Escape)},
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
      {"right", ImGui::GetKeyIndex(ImGuiKey_RightArrow)},
      {"0", GLFW_KEY_0},
      {"1", GLFW_KEY_1},
      {"2", GLFW_KEY_2},
      {"3", GLFW_KEY_3},
      {"4", GLFW_KEY_4},
      {"5", GLFW_KEY_5},
      {"6", GLFW_KEY_6},
      {"7", GLFW_KEY_7},
      {"8", GLFW_KEY_8},
      {"9", GLFW_KEY_9}};

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

ImVec2 GuiWindow::getMouseDelta() { return ImGui::GetIO().MouseDelta; }

ImVec2 GuiWindow::getMouseWheelDelta() { return mMouseWheelDelta; }

ImVec2 GuiWindow::getMousePosition() { return ImGui::GetIO().MousePos; }

bool GuiWindow::isMouseKeyDown(int key) {
  return !ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseDown(key);
}

bool GuiWindow::isMouseKeyClicked(int key) {
  if (!mCursorEnabled) {
    return ImGui::IsMouseClicked(key);
  }
  return !ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseClicked(key);
}

void GuiWindow::setCursorEnabled(bool enabled) {
  mCursorEnabled = enabled;
  if (enabled) {
    glfwSetInputMode(mWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
  } else {
    glfwSetInputMode(mWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouse;
  }
}

bool GuiWindow::getCursorEnabled() const { return mCursorEnabled; }

} // namespace renderer
} // namespace svulkan2
