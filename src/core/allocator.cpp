#include "svulkan2/core/allocator.h"
#include "svulkan2/core/buffer.h"
#include "svulkan2/core/context.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#pragma GCC diagnostic pop

namespace svulkan2 {
namespace core {

Allocator::Allocator(Context &context, VmaAllocatorCreateInfo const &info)
    : mContext(context) {
  if (vmaCreateAllocator(&info, &mMemoryAllocator) != VK_SUCCESS) {
    throw std::runtime_error("failed to create VmaAllocator");
  }
}

std::shared_ptr<class Context> Allocator::getContext() const {
  return mContext.shared_from_this();
}

Allocator::~Allocator() { vmaDestroyAllocator(mMemoryAllocator); }

std::unique_ptr<Buffer> Allocator::allocateStagingBuffer(vk::DeviceSize size, bool readback) {
  if (readback) {
    return std::make_unique<Buffer>(mContext.shared_from_this(), size,
                                    vk::BufferUsageFlagBits::eTransferSrc |
                                    vk::BufferUsageFlagBits::eTransferDst,
                                    VMA_MEMORY_USAGE_GPU_TO_CPU);
  }
  return std::make_unique<Buffer>(mContext.shared_from_this(), size,
                                  vk::BufferUsageFlagBits::eTransferSrc |
                                  vk::BufferUsageFlagBits::eTransferDst,
                                  VMA_MEMORY_USAGE_CPU_ONLY);
}

std::unique_ptr<class Buffer>
Allocator::allocateUniformBuffer(vk::DeviceSize size, bool deviceOnly) {
  if (deviceOnly) {
    return std::make_unique<Buffer>(mContext.shared_from_this(), size,
                                    vk::BufferUsageFlagBits::eUniformBuffer,
                                    VMA_MEMORY_USAGE_GPU_ONLY);
  } else {
    return std::make_unique<Buffer>(mContext.shared_from_this(), size,
                                    vk::BufferUsageFlagBits::eUniformBuffer,
                                    VMA_MEMORY_USAGE_CPU_TO_GPU);
  }
}

} // namespace core
} // namespace svulkan2
