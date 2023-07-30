#include "svulkan2/core/command_pool.h"
#include "svulkan2/core/command_buffer.h"
#include "svulkan2/core/context.h"
#include "svulkan2/core/device.h"
#include "svulkan2/core/physical_device.h"

namespace svulkan2 {
namespace core {

CommandPool::CommandPool(std::shared_ptr<Device> device) : mDevice(device) {
  mPool = device->getInternal().createCommandPoolUnique(
      {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
       device->getPhysicalDevice()->getGraphicsQueueFamilyIndex()});
}

CommandPool::CommandPool() : CommandPool(Context::Get()->getDevice2()) {}

std::unique_ptr<CommandBuffer> CommandPool::allocateCommandBuffer() {
  return std::make_unique<CommandBuffer>(shared_from_this());
}

CommandPool::~CommandPool() {}

} // namespace core
} // namespace svulkan2
