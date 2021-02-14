#pragma once
#include "allocator.h"
#include "svulkan2/common/vk.h"
#include "svulkan2/renderer/gui.h"
#include "svulkan2/resource/manager.h"
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
  vk::UniqueDescriptorPool mDescriptorPool;

  uint32_t mMaxNumMaterials;
  uint32_t mMaxNumTextures;

  std::unique_ptr<resource::SVResourceManager> mResourceManager;

  vk::UniqueDescriptorSetLayout mMetallicDescriptorSetLayout;
  vk::UniqueDescriptorSetLayout mSpecularDescriptorSetLayout;

public:
  Context(uint32_t apiVersion = VK_API_VERSION_1_1, bool present = true,
          uint32_t maxNumMaterials = 5000, uint32_t maxNumTextures = 5000,
          uint32_t defaultMipLevels = 1);
  ~Context();

  vk::Queue getQueue() const;
  inline class Allocator &getAllocator() { return *mAllocator; }

  vk::UniqueCommandBuffer createCommandBuffer(
      vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const;
  void submitCommandBufferAndWait(vk::CommandBuffer commandBuffer) const;
  vk::UniqueFence
  submitCommandBufferForFence(vk::CommandBuffer commandBuffer) const;
  std::future<void> submitCommandBuffer(vk::CommandBuffer commandBuffer) const;

  inline uint32_t getGraphicsQueueFamilyIndex() const {
    return mQueueFamilyIndex;
  }
  inline vk::Instance getInstance() const { return mInstance.get(); }
  inline vk::Device getDevice() const { return mDevice.get(); }
  inline vk::PhysicalDevice getPhysicalDevice() const {
    return mPhysicalDevice;
  }
  inline vk::DescriptorPool getDescriptorPool() const {
    return mDescriptorPool.get();
  }
  inline vk::DescriptorSetLayout getMetallicDescriptorSetLayout() const {
    return mMetallicDescriptorSetLayout.get();
  }
  inline vk::DescriptorSetLayout getSpecularDescriptorSetLayout() const {
    return mSpecularDescriptorSetLayout.get();
  }
  inline resource::SVResourceManager &getResourceManager() const {
    return *mResourceManager;
  }

  std::unique_ptr<renderer::GuiWindow> createWindow(uint32_t width,
                                                    uint32_t height);

private:
  void createInstance();
  void pickSuitableGpuAndQueueFamilyIndex();
  void createDevice();
  void createMemoryAllocator();

  void createCommandPool();
  void createDescriptorPool();
};

} // namespace core
} // namespace svulkan2
