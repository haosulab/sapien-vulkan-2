#pragma once
#include "svulkan2/common/vk.h"
#include <memory>

namespace svulkan2 {
namespace core {

class Instance;
class Device;

class PhysicalDevice : public std::enable_shared_from_this<PhysicalDevice> {
public:
  PhysicalDevice(std::shared_ptr<Instance> instance, std::string const &hint);

  struct DeviceInfo {
    std::string name;
    vk::PhysicalDevice device{};
    bool present{};
    bool supported{};
    int cudaId{-1};
    int pciBus{-1};
    int queueIndex{-1};
    bool rayTracing{};
    int cudaComputeMode{-1};
    bool discrete{};
  };

  inline vk::PhysicalDevice getInternal() const { return mPickedDeviceInfo.device; }
  inline DeviceInfo const &getPickedDeviceInfo() const { return mPickedDeviceInfo; }
  inline vk::PhysicalDeviceLimits const &getPickedDeviceLimits() const {
    return mPickedDeviceLimits;
  }

  vk::PhysicalDeviceRayTracingPipelinePropertiesKHR getRayTracingProperties() const;
  vk::PhysicalDeviceAccelerationStructurePropertiesKHR getASProperties() const;

  inline std::shared_ptr<Instance> getInstance() const { return mInstance; }
  std::shared_ptr<Device> createDevice();

  ~PhysicalDevice();

  PhysicalDevice(PhysicalDevice const &other) = delete;
  PhysicalDevice &operator=(PhysicalDevice const &other) = delete;
  PhysicalDevice(PhysicalDevice const &&other) = delete;
  PhysicalDevice &operator=(PhysicalDevice const &&other) = delete;

  std::vector<DeviceInfo> summarizeDeviceInfo() const;

private:
  std::shared_ptr<Instance> mInstance;
  DeviceInfo mPickedDeviceInfo{};
  vk::PhysicalDeviceLimits mPickedDeviceLimits{};
};

} // namespace core
} // namespace svulkan2
