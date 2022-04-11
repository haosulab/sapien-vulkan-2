#include "svulkan2/core/command_pool.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace core {
CommandPool::CommandPool() {
  mContext = Context::Get();
  mPool = mContext->getDevice().createCommandPoolUnique(
      {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
       mContext->getGraphicsQueueFamilyIndex()});
}

vk::UniqueCommandBuffer
CommandPool::allocateCommandBuffer(vk::CommandBufferLevel level) {
  return std::move(mContext->getDevice()
                       .allocateCommandBuffersUnique({mPool.get(), level, 1})
                       .front());
}

} // namespace core
} // namespace svulkan2
