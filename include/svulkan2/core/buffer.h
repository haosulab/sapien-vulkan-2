#pragma once
#include "svulkan2/common/log.h"
#include "svulkan2/common/vk.h"

#ifdef SVULKAN2_CUDA_INTEROP
#include "svulkan2/common/cuda_helper.h"
#endif

namespace svulkan2 {
namespace core {

class Buffer {
protected:
  std::shared_ptr<class Context> mContext;
  vk::DeviceSize mSize;
  bool mHostVisible;
  bool mHostCoherent;

  vk::Buffer mBuffer;
  VmaAllocation mAllocation;
  VmaAllocationInfo mAllocationInfo;

  bool mMapped{};
  void *mMappedData;

#ifdef TRACK_ALLOCATION
  uint64_t mBufferId{};
#endif

public:
  Buffer(vk::DeviceSize size, vk::BufferUsageFlags usageFlags,
         VmaMemoryUsage memoryUsage,
         VmaAllocationCreateFlags allocationFlags = {});

  Buffer(const Buffer &) = delete;
  Buffer(const Buffer &&) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer &operator=(Buffer &&) = delete;

  ~Buffer();

  vk::Buffer getVulkanBuffer() const { return mBuffer; }

  void *map();
  void unmap();
  void flush();

  void upload(void const *data, size_t size, size_t offset = 0);
  void download(void *data, size_t size, size_t offset = 0);

  template <typename T> void upload(T const &data) {
    upload(&data, sizeof(T), 0);
  }

  template <typename T> void upload(std::vector<T> const &data) {
    upload(data.data(), sizeof(T) * data.size(), 0);
  }

  template <typename T> std::vector<T> download() {
    if ((mSize / sizeof(T)) * sizeof(T) != mSize) {
      throw std::runtime_error(
          "failed to download buffer: incompatible data type");
    }
    std::vector<T> data(mSize / sizeof(T));
    download(data.data(), mSize, 0);
    return data;
  }

  inline vk::DeviceSize getSize() const { return mSize; }

#ifdef SVULKAN2_CUDA_INTEROP

#define checkCudaErrors(call)                                                  \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

private:
  void *mCudaPtr{};
  cudaExternalMemory_t mCudaMem{};
  int mCudaDeviceId{-1};

public:
  void *getCudaPtr();
  int getCudaDeviceId();
#endif
};

} // namespace core
} // namespace svulkan2
