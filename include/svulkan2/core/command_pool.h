#pragma once
#include "svulkan2/common/vk.h"
#include <memory>
#include <mutex>

namespace svulkan2 {
namespace core {
class Context;

class CommandPool {
public:
  vk::UniqueCommandBuffer
  allocateCommandBuffer(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);
  CommandPool();

  CommandPool(CommandPool const &) = delete;
  CommandPool &operator=(CommandPool const &) = delete;
  CommandPool(CommandPool const &&) = delete;
  CommandPool &operator=(CommandPool const &&) = delete;

private:
  std::shared_ptr<Context> mContext;
  vk::UniqueCommandPool mPool;
};

} // namespace core
} // namespace svulkan2
