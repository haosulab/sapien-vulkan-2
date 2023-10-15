#pragma once
#include "svulkan2/common/vk.h"
#include <memory>
#include <mutex>

namespace svulkan2 {
namespace core {
class Context;
class Device;

class CommandPool {
public:
  vk::UniqueCommandBuffer
  allocateCommandBuffer(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);
  CommandPool();

  vk::CommandPool getVulkanCommandPool() const { return mPool.get(); }

private:
  std::shared_ptr<Device> mDevice;
  vk::UniqueCommandPool mPool;
};

} // namespace core
} // namespace svulkan2
