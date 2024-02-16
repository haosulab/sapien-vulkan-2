#pragma once
#include "./instance.h"
#include "svulkan2/common/vk.h"
#include <memory>

namespace svulkan2 {
namespace core {

class Instance;
class Device;

class PhysicalDevice : public std::enable_shared_from_this<PhysicalDevice> {
public:
  PhysicalDevice(std::shared_ptr<Instance> instance, PhysicalDeviceInfo const &deviceInfo);

  inline vk::PhysicalDevice getInternal() const { return mPickedDeviceInfo.device; }
  inline PhysicalDeviceInfo const &getPickedDeviceInfo() const { return mPickedDeviceInfo; }
  inline vk::PhysicalDeviceLimits const &getPickedDeviceLimits() const {
    return mPickedDeviceLimits;
  }

  vk::PhysicalDeviceRayTracingPipelinePropertiesKHR getRayTracingProperties() const;
  vk::PhysicalDeviceAccelerationStructurePropertiesKHR getASProperties() const;

  inline std::shared_ptr<Instance> getInstance() const { return mInstance; }
  std::shared_ptr<Device> createDevice();

  uint32_t getMaxWorkGroupInvocations() const;
  uint32_t getSubgroupSize() const;

  ~PhysicalDevice();

  PhysicalDevice(PhysicalDevice const &other) = delete;
  PhysicalDevice &operator=(PhysicalDevice const &other) = delete;
  PhysicalDevice(PhysicalDevice const &&other) = delete;
  PhysicalDevice &operator=(PhysicalDevice const &&other) = delete;

  // static std::vector<DeviceInfo> summarizeDeviceInfo(Instance const &instance);
  // std::vector<DeviceInfo> summarizeDeviceInfo() const;

private:
  std::shared_ptr<Instance> mInstance;
  PhysicalDeviceInfo mPickedDeviceInfo{};
  vk::PhysicalDeviceLimits mPickedDeviceLimits{};
};

} // namespace core
} // namespace svulkan2
