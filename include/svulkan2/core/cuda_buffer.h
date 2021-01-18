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

public:
  CudaBuffer(
      class Context &context, vk::DeviceSize size,
      vk::BufferUsageFlags usageFlags = vk::BufferUsageFlagBits::eTransferDst);
  inline vk::DeviceSize getSize() const { return mSize; }
};

} // namespace core
} // namespace svulkan2

#endif
