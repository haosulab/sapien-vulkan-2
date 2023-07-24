#include "svulkan2/core/buffer.h"
#include "../common/logger.h"
#include "svulkan2/core/allocator.h"
#include "svulkan2/core/context.h"
#include <easy/profiler.h>

#ifdef SVULKAN2_CUDA_INTEROP
#include "../common/cuda_helper.h"
#endif

namespace svulkan2 {
namespace core {

// TODO: make atomic!!!!!!
#ifdef TRACK_ALLOCATION
static uint64_t gBufferId = 1;
static uint64_t gBufferCount = 0;
#endif

std::unique_ptr<Buffer> Buffer::CreateStaging(vk::DeviceSize size, bool readback) {
  if (readback) {
    return std::make_unique<Buffer>(
        size, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_GPU_TO_CPU);
  }
  return std::make_unique<Buffer>(
      size, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_CPU_ONLY);
}

std::unique_ptr<Buffer> Buffer::CreateUniform(vk::DeviceSize size, bool deviceOnly) {
  if (deviceOnly) {
    return std::make_unique<Buffer>(size, vk::BufferUsageFlagBits::eUniformBuffer,
                                    VMA_MEMORY_USAGE_GPU_ONLY);
  } else {
    return std::make_unique<Buffer>(size, vk::BufferUsageFlagBits::eUniformBuffer,
                                    VMA_MEMORY_USAGE_CPU_TO_GPU);
  }
}

std::unique_ptr<Buffer> Buffer::Create(vk::DeviceSize size, vk::BufferUsageFlags usageFlags,
                                       VmaMemoryUsage memoryUsage,
                                       VmaAllocationCreateFlags allocationFlags, bool external) {
  return std::make_unique<Buffer>(size, usageFlags, memoryUsage, allocationFlags, external);
}

Buffer::Buffer(vk::DeviceSize size, vk::BufferUsageFlags usageFlags, VmaMemoryUsage memoryUsage,
               VmaAllocationCreateFlags allocationFlags, bool external)
    : mSize(size), mExternal(external) {
  mContext = Context::Get();

  vk::BufferCreateInfo bufferInfo({}, size, usageFlags);
  vk::ExternalMemoryBufferCreateInfo externalMemoryBufferInfo(
      vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);

  VmaAllocationCreateInfo memoryInfo{};
  memoryInfo.usage = memoryUsage;
  memoryInfo.flags = allocationFlags;

  if (external) {
    bufferInfo.setPNext(&externalMemoryBufferInfo);
    memoryInfo.pool = mContext->getAllocator().getExternalPool();
  }

  if (vmaCreateBuffer(mContext->getAllocator().getVmaAllocator(),
                      reinterpret_cast<VkBufferCreateInfo *>(&bufferInfo), &memoryInfo,
                      reinterpret_cast<VkBuffer *>(&mBuffer), &mAllocation,
                      &mAllocationInfo) != VK_SUCCESS) {
    throw std::runtime_error("cannot create buffer");
  }

  VkMemoryPropertyFlags memFlags;
  vmaGetMemoryTypeProperties(mContext->getAllocator().getVmaAllocator(),
                             mAllocationInfo.memoryType, &memFlags);
  mHostVisible = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
  mHostCoherent = (memFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

#ifdef TRACK_ALLOCATION
  mBufferId = gBufferId++;
  logger::info("Create Buffer {}; Total {}", mBufferId, ++gBufferCount);
#endif
}

Buffer::~Buffer() {
#ifdef SVULKAN2_CUDA_INTEROP
  if (mCudaPtr) {
    checkCudaErrors(cudaDestroyExternalMemory(mCudaMem));
    checkCudaErrors(cudaFree(mCudaPtr));
  }
#endif
  vmaDestroyBuffer(mContext->getAllocator().getVmaAllocator(), mBuffer, mAllocation);

#ifdef TRACK_ALLOCATION
  mBufferId = gBufferId++;
  logger::info("Destroy Buffer {}, Total {}", mBufferId, --gBufferCount);
#endif
}

void *Buffer::map() {
  if (!mMapped) {
    auto result =
        vmaMapMemory(mContext->getAllocator().getVmaAllocator(), mAllocation, &mMappedData);
    if (result != VK_SUCCESS) {
      logger::critical("unable to map memory");
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
  vmaFlushAllocation(mContext->getAllocator().getVmaAllocator(), mAllocation, 0, mSize);
}

void Buffer::upload(void const *data, size_t size, size_t offset) {
  if (size == 0) {
    return;
  }

  if (offset + size > mSize) {
    throw std::runtime_error("failed to upload buffer: upload size exceeds buffer size");
  }

  if (mHostVisible) {
    if (mMapped) {
      std::memcpy(reinterpret_cast<uint8_t *>(mMappedData) + offset, data, size);
    } else {
      map();
      std::memcpy(reinterpret_cast<uint8_t *>(mMappedData) + offset, data, size);
      unmap();
    }
    if (!mHostCoherent) {
      flush();
    }
  } else {
    auto stagingBuffer = Buffer::CreateStaging(size);
    stagingBuffer->upload(data, size);

    // TODO: find a better way
    auto pool = mContext->createCommandPool();
    auto cb = pool->allocateCommandBuffer();
    cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cb->copyBuffer(stagingBuffer->mBuffer, mBuffer, vk::BufferCopy(0, offset, size));
    cb->end();
    mContext->getQueue().submitAndWait(cb.get());
  }
}

void Buffer::download(void *data, size_t size, size_t offset) {
  if (offset + size > mSize) {
    throw std::runtime_error("failed to download buffer: download size exceeds buffer size");
  }

  if (mHostVisible) {
    map();
    std::memcpy(data, reinterpret_cast<uint8_t *>(mMappedData) + offset, size);
    unmap();
  } else {
    auto stagingBuffer = Buffer::CreateStaging(size);

    auto pool = mContext->createCommandPool();
    auto cb = pool->allocateCommandBuffer();
    cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cb->copyBuffer(mBuffer, stagingBuffer->mBuffer, vk::BufferCopy(offset, 0, size));
    cb->end();
    mContext->getQueue().submitAndWait(cb.get());
    stagingBuffer->download(data, size);
  }
}

vk::DeviceAddress Buffer::getAddress() const {
  return mContext->getDevice().getBufferAddress({mBuffer});
}

#ifdef SVULKAN2_CUDA_INTEROP
void *Buffer::getCudaPtr() {
  if (!mExternal) {
    throw std::runtime_error("failed to get cuda pointer, \"external\" must be "
                             "passed at buffer creation");
  }

  if (mCudaPtr) {
    return mCudaPtr;
  }
  mCudaDeviceId = getCudaDeviceIdFromPhysicalDevice(mContext->getPhysicalDevice());
  if (mCudaDeviceId < 0) {
    throw std::runtime_error(
        "Vulkan Device is not visible to CUDA. You probably need to unset the "
        "CUDA_VISIBLE_DEVICES variable. Or you can try other "
        "CUDA_VISIBLE_DEVICES until you find a working one.");
  }
  checkCudaErrors(cudaSetDevice(mCudaDeviceId));
  cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
  externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  externalMemoryHandleDesc.size = mAllocationInfo.offset + mAllocationInfo.size; // TODO check

  vk::MemoryGetFdInfoKHR vkMemoryGetFdInfoKHR;
  vkMemoryGetFdInfoKHR.setPNext(nullptr);
  vkMemoryGetFdInfoKHR.setMemory(mAllocationInfo.deviceMemory);
  vkMemoryGetFdInfoKHR.setHandleType(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);

  // According to https://docs.nvidia.com/pdf/CUDA_Runtime_API.pdf
  // Performing any operations on the file descriptor after it is imported
  // results in undefined behavior
  auto cudaFd = mContext->getDevice().getMemoryFdKHR(vkMemoryGetFdInfoKHR);
  externalMemoryHandleDesc.handle.fd = cudaFd;

  checkCudaErrors(cudaImportExternalMemory(&mCudaMem, &externalMemoryHandleDesc));

  cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
  externalMemBufferDesc.offset = mAllocationInfo.offset;
  externalMemBufferDesc.size = mAllocationInfo.size;
  externalMemBufferDesc.flags = 0;

  checkCudaErrors(cudaExternalMemoryGetMappedBuffer(&mCudaPtr, mCudaMem, &externalMemBufferDesc));
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
