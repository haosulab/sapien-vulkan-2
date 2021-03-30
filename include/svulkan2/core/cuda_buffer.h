#pragma once
#ifdef CUDA_INTEROP

#include "svulkan2/common/log.h"
#include "svulkan2/common/vk.h"
#include <cuda_runtime.h>

namespace svulkan2 {
namespace core {

class CudaBuffer {
  class Context *mContext;
  vk::DeviceSize mSize;

  vk::UniqueBuffer mBuffer;
  vk::UniqueDeviceMemory mMemory;

  void *mCudaPtr;
  cudaExternalMemory_t mCudaMem;
  int mCudaDeviceId;

public:
  CudaBuffer(
      class Context &context, vk::DeviceSize size,
      vk::BufferUsageFlags usageFlags = vk::BufferUsageFlagBits::eTransferDst);
  ~CudaBuffer();

  inline vk::DeviceSize getSize() const { return mSize; }

  vk::Buffer getVulkanBuffer() const { return mBuffer.get(); }
  vk::DeviceMemory getVulkanMemory() const { return mMemory.get(); }
  void *getCudaPointer() const { return mCudaPtr; }
  int getCudaDeviceId() const { return mCudaDeviceId; }
};

} // namespace core
} // namespace svulkan2

#endif
