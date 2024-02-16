#include "svulkan2/core/command_pool.h"
#include "svulkan2/core/context.h"
#include "svulkan2/core/device.h"

namespace svulkan2 {
namespace core {
CommandPool::CommandPool(std::shared_ptr<Device> device) : mDevice(device) {
  if (!device) {
    throw std::runtime_error("failed to create command pool: invalid device");
  }

  mPool = mDevice->getInternal().createCommandPoolUnique(
      {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
       mDevice->getGraphicsQueueFamilyIndex()});
}

vk::UniqueCommandBuffer CommandPool::allocateCommandBuffer(vk::CommandBufferLevel level) {
  return std::move(
      mDevice->getInternal().allocateCommandBuffersUnique({mPool.get(), level, 1}).front());
}

} // namespace core
} // namespace svulkan2
