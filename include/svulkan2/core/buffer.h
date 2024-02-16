#pragma once
#include "svulkan2/common/vk.h"
#include <memory>

typedef struct CUexternalMemory_st *cudaExternalMemory_t;

namespace svulkan2 {
namespace core {

class Device;

class Buffer {
public:
  static std::unique_ptr<Buffer> CreateStaging(vk::DeviceSize size, bool readback = false);
  static std::unique_ptr<Buffer> CreateUniform(vk::DeviceSize size, bool deviceOnly = false,
                                               bool external = false);
  static std::unique_ptr<Buffer> Create(vk::DeviceSize size, vk::BufferUsageFlags usageFlags,
                                        VmaMemoryUsage memoryUsage,
                                        VmaAllocationCreateFlags allocationFlags = {},
                                        bool external = false, VmaPool pool = {});

public:
  Buffer(std::shared_ptr<Device> device, vk::DeviceSize size, vk::BufferUsageFlags usageFlags,
         VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags allocationFlags = {},
         bool external = false, VmaPool pool = {});

  Buffer(const Buffer &) = delete;
  Buffer(const Buffer &&) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer &operator=(Buffer &&) = delete;

  vk::Buffer getVulkanBuffer() const { return mBuffer; }
  vk::DeviceAddress getAddress() const;

  void *map();
  void unmap();
  void flush();

  void upload(void const *data, size_t size, size_t offset = 0);
  void download(void *data, size_t size, size_t offset = 0);

  template <typename T> void upload(T const &data) { upload(&data, sizeof(T), 0); }

  template <typename T> void upload(std::vector<T> const &data) {
    upload(data.data(), sizeof(T) * data.size(), 0);
  }

  template <typename T> std::vector<T> download() {
    if ((mSize / sizeof(T)) * sizeof(T) != mSize) {
      throw std::runtime_error("failed to download buffer: incompatible data type");
    }
    std::vector<T> data(mSize / sizeof(T));
    download(data.data(), mSize, 0);
    return data;
  }

  inline vk::DeviceSize getSize() const { return mSize; }

  ~Buffer();

protected:
  std::shared_ptr<Device> mDevice;
  vk::DeviceSize mSize{};
  bool mHostVisible{};
  bool mHostCoherent{};

  vk::Buffer mBuffer{};
  VmaAllocation mAllocation;
  VmaAllocationInfo mAllocationInfo;

  bool mMapped{};
  void *mMappedData{};

  bool mExternal{};

#ifdef SVULKAN2_CUDA_INTEROP

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
