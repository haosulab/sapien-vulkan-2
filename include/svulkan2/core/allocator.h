#pragma once
#include "svulkan2/common/vk.h"
#include <memory>

namespace svulkan2 {
namespace core {

class Allocator {
  VmaAllocator mMemoryAllocator;
  VmaPool mExternalMemoryPool;
  vk::ExportMemoryAllocateInfo mExternalAllocInfo;

public:
  Allocator(VmaAllocatorCreateInfo const &info);
  ~Allocator();
  Allocator(Allocator &other) = delete;
  Allocator(Allocator &&other) = delete;
  Allocator &operator=(Allocator &other) = delete;
  Allocator &operator=(Allocator &&other) = delete;

  VmaAllocator getVmaAllocator() const { return mMemoryAllocator; }
  VmaPool getExternalPool() const { return mExternalMemoryPool; };

public:
  std::unique_ptr<class Buffer> allocateStagingBuffer(vk::DeviceSize size,
                                                      bool readback = false);
  std::unique_ptr<class Buffer> allocateUniformBuffer(vk::DeviceSize size,
                                                      bool deviceOnly = false);
};

} // namespace core
} // namespace svulkan2
