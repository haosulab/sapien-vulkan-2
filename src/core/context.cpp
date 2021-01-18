#include "svulkan2/core/context.h"
#include "svulkan2/common/log.h"
#include <GLFW/glfw3.h>

#include "svulkan2/core/allocator.h"

namespace svulkan2 {
namespace core {

static void glfwErrorCallback(int error_code, const char *description) {
  log::error("GLFW error: {}", description);
}

#ifdef VK_VALIDATION
static bool checkValidationLayerSupport() {
  static const std::vector<const char *> validationLayers = {
      "VK_LAYER_KHRONOS_validation"};
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
  for (const char *layerName : validationLayers) {
    bool layerFound = false;

    for (const auto &layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }
    if (!layerFound) {
      return false;
    }
  }
  return true;
}
#endif

Context::Context(uint32_t apiVersion, bool present)
    : mApiVersion(apiVersion), mPresent(present) {
  createInstance();
  pickSuitableGpuAndQueueFamilyIndex();
  createDevice();
  createMemoryAllocator();
  createCommandPool();
}

Context::~Context() {}

void Context::createInstance() {
  if (mPresent) {
    log::info("Initializing GLFW");
    glfwInit();
    glfwSetErrorCallback(glfwErrorCallback);
  }

#ifdef VK_VALIDATION
  if (!checkValidationLayerSupport()) {
    throw std::runtime_error(
        "createInstance: validation layers are not available");
  }
  std::vector<const char *> enabledLayers = {"VK_LAYER_KHRONOS_validation"};
#else
  std::vector<const char *> enabledLayers = {};
#endif
  vk::ApplicationInfo appInfo("Vulkan Renderer", VK_MAKE_VERSION(0, 0, 1),
                              "No Engine", VK_MAKE_VERSION(0, 0, 1),
                              mApiVersion);

  std::vector<const char *> instanceExtensions;
  if (mPresent) {
    if (!glfwVulkanSupported()) {
      throw std::runtime_error("createInstance: GLFW does not support Vulkan");
    }
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (!glfwExtensions) {
      throw std::runtime_error(
          "createInstance: Vulkan does not support GLFW extensions");
    }
    for (uint32_t i = 0; i < glfwExtensionCount; ++i) {
      instanceExtensions.push_back(glfwExtensions[i]);
    }
  }
  vk::InstanceCreateInfo createInfo(
      {}, &appInfo, enabledLayers.size(), enabledLayers.data(),
      instanceExtensions.size(), instanceExtensions.data());
  mInstance = vk::createInstanceUnique(createInfo);
}

void Context::pickSuitableGpuAndQueueFamilyIndex() {
  GLFWwindow *window;
  VkSurfaceKHR tmpSurface;
  if (mPresent) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(1, 1, "vulkan", nullptr, nullptr);
    auto result =
        glfwCreateWindowSurface(mInstance.get(), window, nullptr, &tmpSurface);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Window creation failed, you may not create "
                               "GLFW window for presenting rendered results.");
    }
  }

  vk::PhysicalDevice pickedDevice;
  uint32_t pickedIndex;
  for (auto device : mInstance->enumeratePhysicalDevices()) {
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
        device.getQueueFamilyProperties();
    pickedIndex = queueFamilyProperties.size();
    for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
      if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
        if (mPresent && !device.getSurfaceSupportKHR(i, tmpSurface)) {
          continue;
        }
        pickedIndex = i;
      }
    }
    if (pickedIndex == queueFamilyProperties.size()) {
      continue;
    }
    auto features = device.getFeatures();
    if (!features.independentBlend) {
      continue;
    }
    pickedDevice = device;
    break;
  }
  if (!pickedDevice) {
    throw std::runtime_error(
        "pickSuitableGpuAndQueue: no compatible GPU found");
  }
  if (mPresent) {
    vkDestroySurfaceKHR(mInstance.get(), tmpSurface, nullptr);
    glfwDestroyWindow(window);
  }
  mPhysicalDevice = pickedDevice;
  mQueueFamilyIndex = pickedIndex;
}

void Context::createDevice() {
  float queuePriority = 0.0f;
  vk::DeviceQueueCreateInfo deviceQueueCreateInfo({}, mQueueFamilyIndex, 1,
                                                  &queuePriority);
  std::vector<const char *> deviceExtensions{};
  vk::PhysicalDeviceFeatures features;
  features.independentBlend = true;

  if (mPresent) {
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }
  mDevice = mPhysicalDevice.createDeviceUnique(vk::DeviceCreateInfo(
      {}, 1, &deviceQueueCreateInfo, 0, nullptr, deviceExtensions.size(),
      deviceExtensions.data(), &features));
}

void Context::createMemoryAllocator() {
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.vulkanApiVersion = mApiVersion;
  allocatorInfo.physicalDevice = mPhysicalDevice;
  allocatorInfo.device = mDevice.get();
  allocatorInfo.instance = mInstance.get();
  mAllocator = std::make_unique<Allocator>(*this, allocatorInfo);
}

void Context::createCommandPool() {
  mCommandPool = mDevice->createCommandPoolUnique(vk::CommandPoolCreateInfo(
      vk::CommandPoolCreateFlagBits::eResetCommandBuffer, mQueueFamilyIndex));
}

vk::UniqueCommandBuffer
Context::createCommandBuffer(vk::CommandBufferLevel level) const {
  return std::move(
      mDevice->allocateCommandBuffersUnique({mCommandPool.get(), level, 1})
          .front());
}

void Context::submitCommandBufferAndWait(
    vk::CommandBuffer commandBuffer) const {
  auto fence = mDevice->createFenceUnique({});
  getQueue().submit(vk::SubmitInfo(0, nullptr, nullptr, 1, &commandBuffer),
                    fence.get());
  mDevice->waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
}

vk::UniqueFence
Context::submitCommandBufferForFence(vk::CommandBuffer commandBuffer) const {
  auto fence = mDevice->createFenceUnique({});
  getQueue().submit(vk::SubmitInfo(0, nullptr, nullptr, 1, &commandBuffer),
                    fence.get());
  return fence;
}

std::future<void>
Context::submitCommandBuffer(vk::CommandBuffer commandBuffer) const {
  auto fence = mDevice->createFenceUnique({});
  getQueue().submit(vk::SubmitInfo(0, nullptr, nullptr, 1, &commandBuffer),
                    fence.get());
  return std::async(std::launch::async, [fence = std::move(fence), this]() {
    mDevice->waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
  });
}

vk::Queue Context::getQueue() const {
  return mDevice->getQueue(mQueueFamilyIndex, 0);
}

} // namespace core

} // namespace svulkan2
