#include "svulkan2/core/buffer.h"
#include "svulkan2/core/allocator.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace core {
Buffer::Buffer(Context &context, vk::DeviceSize size,
               vk::BufferUsageFlags usageFlags, VmaMemoryUsage memoryUsage,
               VmaAllocationCreateFlags allocationFlags)
    : mContext(&context), mSize(size) {

  vk::BufferCreateInfo bufferInfo({}, size, usageFlags);

  VmaAllocationCreateInfo memoryInfo{};
  memoryInfo.usage = memoryUsage;
  memoryInfo.flags = allocationFlags;

  VmaAllocationInfo allocInfo;

  if (vmaCreateBuffer(mContext->getAllocator().getVmaAllocator(),
                      reinterpret_cast<VkBufferCreateInfo *>(&bufferInfo),
                      &memoryInfo, reinterpret_cast<VkBuffer *>(&mBuffer),
                      &mAllocation, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("cannot create image");
  }

  VkMemoryPropertyFlags memFlags;
  vmaGetMemoryTypeProperties(context.getAllocator().getVmaAllocator(),
                             allocInfo.memoryType, &memFlags);
  mHostVisible = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
  mHostVisible = (memFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
}

Buffer::~Buffer() {
  vmaDestroyBuffer(mContext->getAllocator().getVmaAllocator(), mBuffer,
                   mAllocation);
}

void *Buffer::map() {
  if (!mMapped) {
    auto result = vmaMapMemory(mContext->getAllocator().getVmaAllocator(),
                               mAllocation, &mMappedData);
    if (result != VK_SUCCESS) {
      log::critical("unable to map memory");
      abort();
    }
  }
  return mMappedData;
}

void Buffer::unmap() {
  if (mMapped) {
    vmaUnmapMemory(mContext->getAllocator().getVmaAllocator(), mAllocation);
    mMapped = false;
  }
}

void Buffer::flush() {
  vmaFlushAllocation(mContext->getAllocator().getVmaAllocator(), mAllocation, 0,
                     mSize);
}

void Buffer::upload(void const *data, size_t size, size_t offset) {
  if (offset + size > mSize) {
    throw std::runtime_error("buffer upload size exceeds buffer size");
  }

  if (mHostVisible) {
    map();
    std::memcpy(reinterpret_cast<uint8_t *>(mMappedData) + offset, data, size);
    unmap();
    if (!mHostCoherent) {
      flush();
    }
  } else {
    auto stagingBuffer = mContext->getAllocator().allocateStagingBuffer(size);
    stagingBuffer->upload(data, size);
    auto cb = mContext->createCommandBuffer();
    cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cb->copyBuffer(stagingBuffer->mBuffer, mBuffer,
                   vk::BufferCopy(0, offset, size));
    cb->end();
    mContext->submitCommandBuffer(cb.get());
  }
}

} // namespace core
} // namespace svulkan2
