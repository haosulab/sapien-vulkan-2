#include "svulkan2/core/allocator.h"
#include "svulkan2/core/buffer.h"
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

namespace svulkan2 {
namespace core {

Allocator::Allocator(Context &context, VmaAllocatorCreateInfo const &info)
    : mContext(&context) {
  if (vmaCreateAllocator(&info, &mMemoryAllocator) != VK_SUCCESS) {
    throw std::runtime_error("failed to create VmaAllocator");
  }
}

Allocator::~Allocator() { vmaDestroyAllocator(mMemoryAllocator); }

std::unique_ptr<Buffer> Allocator::allocateStagingBuffer(vk::DeviceSize size) {
  return std::make_unique<Buffer>(*mContext, size,
                                  vk::BufferUsageFlagBits::eTransferSrc,
                                  VMA_MEMORY_USAGE_CPU_COPY);
}

std::unique_ptr<class Buffer>
Allocator::allocateUniformBuffer(vk::DeviceSize size, bool deviceOnly) {
  if (deviceOnly) {
    return std::make_unique<Buffer>(*mContext, size,
                                    vk::BufferUsageFlagBits::eUniformBuffer,
                                    VMA_MEMORY_USAGE_GPU_ONLY);
  } else {
    return std::make_unique<Buffer>(*mContext, size,
                                    vk::BufferUsageFlagBits::eUniformBuffer,
                                    VMA_MEMORY_USAGE_CPU_TO_GPU);
  }
}

} // namespace core
} // namespace svulkan2
