#include "svulkan2/core/command_pool.h"
#include "svulkan2/core/context.h"
#include "svulkan2/core/device.h"

namespace svulkan2 {
namespace core {
CommandPool::CommandPool() {
  auto context = Context::Get();
  mDevice = context->getDevice2();
  mPool = mDevice->getInternal().createCommandPoolUnique(
      {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
       context->getGraphicsQueueFamilyIndex()});
}

vk::UniqueCommandBuffer CommandPool::allocateCommandBuffer(vk::CommandBufferLevel level) {
  return std::move(
      mDevice->getInternal().allocateCommandBuffersUnique({mPool.get(), level, 1}).front());
}

} // namespace core
} // namespace svulkan2
