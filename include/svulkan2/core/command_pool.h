#pragma once
#include "svulkan2/common/vk.h"
#include <memory>
#include <mutex>

namespace svulkan2 {
namespace core {
class Device;
class CommandBuffer;

class CommandPool : public std::enable_shared_from_this<CommandPool> {
public:
  std::unique_ptr<CommandBuffer>
  allocateCommandBuffer(); // TODO: support secondary command buffers

  CommandPool(std::shared_ptr<Device>);

  inline std::shared_ptr<Device> getDevice() const { return mDevice; }
  inline vk::CommandPool getInternal() const { return mPool.get(); }

  // construct from global context
  CommandPool();

  CommandPool(CommandPool const &) = delete;
  CommandPool &operator=(CommandPool const &) = delete;
  CommandPool(CommandPool const &&) = delete;
  CommandPool &operator=(CommandPool const &&) = delete;
  ~CommandPool();

private:
  std::shared_ptr<Device> mDevice;
  vk::UniqueCommandPool mPool;
};

} // namespace core
} // namespace svulkan2
