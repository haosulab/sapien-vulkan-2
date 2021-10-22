#include "svulkan2/core/buffer.h"
#include "svulkan2/core/allocator.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace core {

Buffer::Buffer(vk::DeviceSize size, vk::BufferUsageFlags usageFlags,
               VmaMemoryUsage memoryUsage,
               VmaAllocationCreateFlags allocationFlags)
    : mSize(size) {
  mContext = Context::Get();

  vk::BufferCreateInfo bufferInfo({}, size, usageFlags);

  VmaAllocationCreateInfo memoryInfo{};
  memoryInfo.usage = memoryUsage;
  memoryInfo.flags = allocationFlags;

  VmaAllocationInfo allocInfo;
  if (vmaCreateBuffer(mContext->getAllocator().getVmaAllocator(),
                      reinterpret_cast<VkBufferCreateInfo *>(&bufferInfo),
                      &memoryInfo, reinterpret_cast<VkBuffer *>(&mBuffer),
                      &mAllocation, &allocInfo) != VK_SUCCESS) {
    throw std::runtime_error("cannot create buffer");
  }

  VkMemoryPropertyFlags memFlags;
  vmaGetMemoryTypeProperties(mContext->getAllocator().getVmaAllocator(),
                             allocInfo.memoryType, &memFlags);
  mHostVisible = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
  mHostCoherent = (memFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
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
    mMapped = true;
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
    throw std::runtime_error(
        "failed to upload bfufer: upload size exceeds buffer size");
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
    mContext->submitCommandBufferAndWait(cb.get());
  }
}

void Buffer::download(void *data, size_t size, size_t offset) {
  if (offset + size > mSize) {
    throw std::runtime_error(
        "failed to download buffer: download size exceeds buffer size");
  }

  if (mHostVisible) {
    map();
    std::memcpy(data, reinterpret_cast<uint8_t *>(mMappedData) + offset, size);
    unmap();
  } else {
    auto stagingBuffer = mContext->getAllocator().allocateStagingBuffer(size);
    auto cb = mContext->createCommandBuffer();
    cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cb->copyBuffer(mBuffer, stagingBuffer->mBuffer,
                   vk::BufferCopy(offset, 0, size));
    cb->end();
    mContext->submitCommandBufferAndWait(cb.get());
    stagingBuffer->download(data, size);
  }
}

} // namespace core
} // namespace svulkan2
