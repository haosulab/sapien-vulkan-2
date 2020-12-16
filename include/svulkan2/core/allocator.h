#pragma once
#include "svulkan2/common/vk.h"
#include <memory>

namespace svulkan2 {
namespace core {

class Allocator {
  class Context *mContext;
  VmaAllocator mMemoryAllocator;

public:
  Allocator(class Context &context, VmaAllocatorCreateInfo const &info);
  ~Allocator();
  Allocator(Allocator &other) = delete;
  Allocator(Allocator &&other) = default;
  Allocator &operator=(Allocator &other) = delete;
  Allocator &operator=(Allocator &&other) = default;

  inline Context &getContext() const { return *mContext; }
  VmaAllocator getVmaAllocator() const { return mMemoryAllocator; }

public:
  std::unique_ptr<class Buffer> allocateStagingBuffer(vk::DeviceSize size);
  std::unique_ptr<class Buffer> allocateUniformBuffer(vk::DeviceSize size,
                                                      bool deviceOnly = false);
};

} // namespace core
} // namespace svulkan2
