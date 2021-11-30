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

  if (vmaCreateBuffer(mContext->getAllocator().getVmaAllocator(),
                      reinterpret_cast<VkBufferCreateInfo *>(&bufferInfo),
                      &memoryInfo, reinterpret_cast<VkBuffer *>(&mBuffer),
                      &mAllocation, &mAllocationInfo) != VK_SUCCESS) {
    throw std::runtime_error("cannot create buffer");
  }

  VkMemoryPropertyFlags memFlags;
  vmaGetMemoryTypeProperties(mContext->getAllocator().getVmaAllocator(),
                             mAllocationInfo.memoryType, &memFlags);
  mHostVisible = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
  mHostCoherent = (memFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
}

Buffer::~Buffer() {
#ifdef CUDA_INTEROP
  if (mCudaPtr) {
    checkCudaErrors(cudaDestroyExternalMemory(mCudaMem));
    checkCudaErrors(cudaFree(mCudaPtr));
    close(mCudaFd);
  }
#endif
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
        "failed to upload buffer: upload size exceeds buffer size");
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

#ifdef CUDA_INTEROP
void *Buffer::getCudaPtr() {
  if (mCudaPtr) {
    return mCudaPtr;
  }
  mCudaDeviceId =
      getCudaDeviceIdFromPhysicalDevice(mContext->getPhysicalDevice());
  if (mCudaDeviceId < 0) {
    throw std::runtime_error(
        "Vulkan Device is not visible to CUDA. You probably need to unset the "
        "CUDA_VISIBLE_DEVICES variable. Or you can try other "
        "CUDA_VISIBLE_DEVICES until you find a working one.");
  }
  checkCudaErrors(cudaSetDevice(mCudaDeviceId));
  cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
  externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  externalMemoryHandleDesc.size =
      mAllocationInfo.offset + mAllocationInfo.size; // TODO check

  vk::MemoryGetFdInfoKHR vkMemoryGetFdInfoKHR;
  vkMemoryGetFdInfoKHR.setPNext(nullptr);
  vkMemoryGetFdInfoKHR.setMemory(mAllocationInfo.deviceMemory);
  vkMemoryGetFdInfoKHR.setHandleType(
      vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
  mCudaFd = mContext->getDevice().getMemoryFdKHR(vkMemoryGetFdInfoKHR);

  externalMemoryHandleDesc.handle.fd = mCudaFd;
  checkCudaErrors(
      cudaImportExternalMemory(&mCudaMem, &externalMemoryHandleDesc));

  cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
  externalMemBufferDesc.offset = mAllocationInfo.offset;
  externalMemBufferDesc.size = mAllocationInfo.size;
  externalMemBufferDesc.flags = 0;

  checkCudaErrors(cudaExternalMemoryGetMappedBuffer(&mCudaPtr, mCudaMem,
                                                    &externalMemBufferDesc));
  return mCudaPtr;
}

int Buffer::getCudaDeviceId() {
  if (!mCudaPtr) {
    return getCudaDeviceIdFromPhysicalDevice(mContext->getPhysicalDevice());
  }
  return mCudaDeviceId;
}
#endif

} // namespace core
} // namespace svulkan2
