#pragma once
#include "command_pool.h"
#include "svulkan2/common/vk.h"
#include <memory>

namespace svulkan2 {
namespace core {
class Instance;
class PhysicalDevice;
class Queue;
class Allocator;

class Device : public std::enable_shared_from_this<Device> {
public:
  Device(std::shared_ptr<PhysicalDevice> physicalDevice);
  inline std::shared_ptr<PhysicalDevice> getPhysicalDevice() const { return mPhysicalDevice; }
  inline vk::Device getInternal() const { return mDevice.get(); }
  inline Queue &getQueue() const { return *mQueue; }
  inline Allocator &getAllocator() const { return *mAllocator; }

  uint32_t getGraphicsQueueFamilyIndex() const;

  std::unique_ptr<CommandPool> createCommandPool();

  ~Device();

  Device(Device const &other) = delete;
  Device &operator=(Device const &other) = delete;
  Device(Device const &&other) = delete;
  Device &operator=(Device const &&other) = delete;

private:
  std::shared_ptr<PhysicalDevice> mPhysicalDevice;
  vk::UniqueDevice mDevice;
  std::unique_ptr<Queue> mQueue;
  std::unique_ptr<Allocator> mAllocator;
};

} // namespace core
} // namespace svulkan2
