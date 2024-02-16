#include "svulkan2/core/device.h"
#include "svulkan2/core/allocator.h"
#include "svulkan2/core/instance.h"
#include "svulkan2/core/physical_device.h"
#include "svulkan2/core/queue.h"

namespace svulkan2 {
namespace core {

Device::Device(std::shared_ptr<PhysicalDevice> physicalDevice) : mPhysicalDevice(physicalDevice) {
  if (!physicalDevice) {
    throw std::runtime_error("failed to create device: invalid physical device");
  }

  float queuePriority = 0.0f;
  int queueIndex = mPhysicalDevice->getPickedDeviceInfo().queueIndex;
  vk::DeviceQueueCreateInfo deviceQueueCreateInfo({}, queueIndex, 1, &queuePriority);
  std::vector<const char *> deviceExtensions{};

  vk::PhysicalDeviceFeatures2 features{};
  vk::PhysicalDeviceDescriptorIndexingFeatures descriptorFeatures{};
  vk::PhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures{};

  features.setPNext(&descriptorFeatures);
  descriptorFeatures.setPNext(&timelineSemaphoreFeatures);

  features.features.setIndependentBlend(true);
  features.features.setWideLines(true);
  features.features.setGeometryShader(true);
  descriptorFeatures.setDescriptorBindingPartiallyBound(true);
  timelineSemaphoreFeatures.setTimelineSemaphore(true);
  descriptorFeatures.setPNext(&timelineSemaphoreFeatures);

  vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extendedDynamicStateFeature;
  extendedDynamicStateFeature.setExtendedDynamicState(true);
  timelineSemaphoreFeatures.setPNext(&extendedDynamicStateFeature);

  vk::PhysicalDeviceAccelerationStructureFeaturesKHR asFeature;
  asFeature.setAccelerationStructure(true);
  vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rtFeature;
  rtFeature.setRayTracingPipeline(true);
  vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR addrFeature;
  addrFeature.setBufferDeviceAddress(true);
  vk::PhysicalDeviceShaderClockFeaturesKHR clockFeature;
  clockFeature.setShaderDeviceClock(true);
  clockFeature.setShaderSubgroupClock(true);

  if (mPhysicalDevice->getPickedDeviceInfo().rayTracing) {
    deviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
    descriptorFeatures.setRuntimeDescriptorArray(true);
    descriptorFeatures.setShaderStorageBufferArrayNonUniformIndexing(true);
    descriptorFeatures.setShaderSampledImageArrayNonUniformIndexing(true);
    extendedDynamicStateFeature.setPNext(&asFeature);
    asFeature.setPNext(&rtFeature);
    rtFeature.setPNext(&addrFeature);
    addrFeature.setPNext(&clockFeature);
    features.features.setShaderInt64(true);
  }

  deviceExtensions.push_back(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);

#ifdef SVULKAN2_CUDA_INTEROP
  deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
  deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
  deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
  deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif

  if (mPhysicalDevice->getPickedDeviceInfo().present &&
      mPhysicalDevice->getInstance()->isGLFWEnabled()) {
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  vk::DeviceCreateInfo deviceInfo({}, deviceQueueCreateInfo, {}, deviceExtensions);
  deviceInfo.setPNext(&features);
  mDevice = mPhysicalDevice->getPickedDeviceInfo().device.createDeviceUnique(deviceInfo);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(mDevice.get());

  mQueue = std::make_unique<Queue>(*this, queueIndex);

  auto instance = mPhysicalDevice->getInstance();
  mAllocator = std::make_unique<Allocator>(*this);
}

uint32_t Device::getGraphicsQueueFamilyIndex() const {
  return mPhysicalDevice->getPickedDeviceInfo().queueIndex;
}

std::unique_ptr<CommandPool> Device::createCommandPool() {
  return std::make_unique<CommandPool>(shared_from_this());
}

Device::~Device(){};

} // namespace core
} // namespace svulkan2
