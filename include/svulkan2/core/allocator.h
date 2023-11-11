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
  VmaPool getRTPool() const {
    if (mRTPool) {
      return mRTPool;
    }
    throw std::runtime_error("this physical device does not support ray tracing");
  };

  ~Allocator();

private:
  VmaAllocator mMemoryAllocator;
  VmaPool mExternalMemoryPool{};
  vk::ExportMemoryAllocateInfo mExternalAllocInfo;

  // AS & SBT in ray tracing requires special alignment
  // we allocate them from this pool
  VmaPool mRTPool{};
};

} // namespace core
} // namespace svulkan2
