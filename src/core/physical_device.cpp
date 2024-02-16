#include "svulkan2/core/physical_device.h"
#include "../common/cuda_helper.h"
#include "../common/logger.h"
#include "svulkan2/core/device.h"
#include <GLFW/glfw3.h>
#include <iomanip>
#include <sstream>

namespace svulkan2 {
namespace core {

PhysicalDevice::PhysicalDevice(std::shared_ptr<Instance> instance,
                               PhysicalDeviceInfo const &deviceInfo)
    : mInstance(instance), mPickedDeviceInfo(deviceInfo) {
  mPickedDeviceLimits = mPickedDeviceInfo.device.getProperties().limits;
}

std::shared_ptr<Device> PhysicalDevice::createDevice() {
  return std::make_shared<Device>(shared_from_this());
}

vk::PhysicalDeviceRayTracingPipelinePropertiesKHR PhysicalDevice::getRayTracingProperties() const {
  if (!mPickedDeviceInfo.rayTracing) {
    throw std::runtime_error("the physical device does not support ray tracing");
  }

  auto properties =
      mPickedDeviceInfo.device.getProperties2<vk::PhysicalDeviceProperties2,
                                              vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
  return properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
}

vk::PhysicalDeviceAccelerationStructurePropertiesKHR PhysicalDevice::getASProperties() const {
  if (!mPickedDeviceInfo.rayTracing) {
    throw std::runtime_error("the physical device does not support ray tracing");
  }
  auto properties = mPickedDeviceInfo.device.getProperties2<
      vk::PhysicalDeviceProperties2, vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
  return properties.get<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
}

uint32_t PhysicalDevice::getMaxWorkGroupInvocations() const {
  return std::min(mPickedDeviceLimits.maxComputeWorkGroupInvocations,
                  mPickedDeviceInfo.subgroupSize * mPickedDeviceInfo.subgroupSize);
}
uint32_t PhysicalDevice::getSubgroupSize() const { return mPickedDeviceInfo.subgroupSize; }

PhysicalDevice::~PhysicalDevice() {}

} // namespace core
} // namespace svulkan2
