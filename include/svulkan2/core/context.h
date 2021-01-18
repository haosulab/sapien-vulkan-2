#pragma once
#include "svulkan2/common/vk.h"
#include "allocator.h"
#include <future>

namespace svulkan2 {
namespace core {

class Context {
  uint32_t mApiVersion;
  bool mPresent;

  vk::UniqueInstance mInstance;

  vk::PhysicalDevice mPhysicalDevice;
  uint32_t mQueueFamilyIndex;

  vk::UniqueDevice mDevice;
  std::unique_ptr<class Allocator> mAllocator;

  vk::UniqueCommandPool mCommandPool;

public:
  Context(uint32_t apiVersion = VK_API_VERSION_1_1, bool present = true);
  ~Context();

  vk::Queue getQueue() const;
  inline class Allocator &getAllocator() { return *mAllocator; }

  vk::UniqueCommandBuffer createCommandBuffer(
      vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const;
  void submitCommandBufferAndWait(vk::CommandBuffer commandBuffer) const;
  vk::UniqueFence
  submitCommandBufferForFence(vk::CommandBuffer commandBuffer) const;
  std::future<void> submitCommandBuffer(vk::CommandBuffer commandBuffer) const;

  vk::Device getDevice() const { return mDevice.get(); }
  vk::PhysicalDevice getPhysicalDevice() const { return mPhysicalDevice; }

private:
  void createInstance();
  void pickSuitableGpuAndQueueFamilyIndex();
  void createDevice();
  void createMemoryAllocator();

  void createCommandPool();
};

} // namespace core
} // namespace svulkan2
