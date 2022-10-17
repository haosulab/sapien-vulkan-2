#pragma once
#include "svulkan2/common/vk.h"
#include <mutex>
#include <memory>

namespace svulkan2 {
namespace core {
class Context;

class CommandPool {
  std::shared_ptr<Context> mContext;
  vk::UniqueCommandPool mPool;

public:
  vk::UniqueCommandBuffer allocateCommandBuffer(
      vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);
  CommandPool();
};

} // namespace core
} // namespace svulkan2
