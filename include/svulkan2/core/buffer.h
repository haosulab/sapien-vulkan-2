#pragma once
#include "svulkan2/common/log.h"
#include "svulkan2/common/vk.h"

namespace svulkan2 {
namespace core {

class Buffer {
  class Context *mContext;
  vk::DeviceSize mSize;
  bool mHostVisible;
  bool mHostCoherent;

  vk::Buffer mBuffer;
  VmaAllocation mAllocation;

  bool mMapped;
  void *mMappedData;

public:
  Buffer(class Context &context, vk::DeviceSize size,
         vk::BufferUsageFlags usageFlags, VmaMemoryUsage memoryUsage,
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

  template <typename T> void upload(T const &data) {
    upload(&data, sizeof(T), 0);
  }

  template <typename T> void upload(std::vector<T> const &data) {
    upload(data.data(), sizeof(T) * data.size(), 0);
  }

  inline vk::DeviceSize getSize() const { return mSize; }
};

} // namespace core
} // namespace svulkan2
