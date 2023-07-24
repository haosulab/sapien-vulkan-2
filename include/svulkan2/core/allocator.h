#pragma once
#include "svulkan2/common/vk.h"
#include <memory>

namespace svulkan2 {
namespace core {

class Device;

class Allocator {
public:
  Allocator(Device &device);
  Allocator(Allocator &other) = delete;
  Allocator(Allocator &&other) = delete;
  Allocator &operator=(Allocator &other) = delete;
  Allocator &operator=(Allocator &&other) = delete;

  VmaAllocator getVmaAllocator() const { return mMemoryAllocator; }
  VmaPool getExternalPool() const { return mExternalMemoryPool; };

  ~Allocator();

private:
  VmaAllocator mMemoryAllocator;
  VmaPool mExternalMemoryPool;
  vk::ExportMemoryAllocateInfo mExternalAllocInfo;
};

} // namespace core
} // namespace svulkan2
