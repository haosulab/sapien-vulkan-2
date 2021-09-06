#pragma once
#include "allocator.h"
#include "svulkan2/common/vk.h"
#include "svulkan2/renderer/gui.h"
#include "svulkan2/resource/manager.h"
#include <future>

namespace svulkan2 {
namespace core {

class Context : public std::enable_shared_from_this<Context> {
  uint32_t mApiVersion;
  bool mVulkanAvailable;
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
  uint32_t mDefaultMipLevels;

  std::string mDeviceHint;

  std::weak_ptr<resource::SVResourceManager> mResourceManager;

  vk::UniqueDescriptorSetLayout mMetallicDescriptorSetLayout;

public:
  static std::shared_ptr<Context> Create(bool present = true,
                                         uint32_t maxNumMaterials = 5000,
                                         uint32_t maxNumTextures = 5000,
                                         uint32_t defaultMipLevels = 1);

  Context(bool present = true, uint32_t maxNumMaterials = 5000,
          uint32_t maxNumTextures = 5000, uint32_t defaultMipLevels = 1, std::string device = "");
  ~Context();

  inline bool isVulkanAvailable() const { return mVulkanAvailable; }
  inline bool isPresentAvailable() const { return mPresent; }

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

  std::shared_ptr<resource::SVResourceManager> getResourceManager() const;
  std::shared_ptr<resource::SVResourceManager> createResourceManager();

  std::unique_ptr<renderer::GuiWindow> createWindow(uint32_t width,
                                                    uint32_t height);

  std::shared_ptr<resource::SVMetallicMaterial>
  createMetallicMaterial(glm::vec4 baseColor, float fresnel, float roughness,
                         float metallic, float transparency);

  std::shared_ptr<resource::SVModel> createModel(
      std::vector<std::shared_ptr<resource::SVMesh>> const &meshes,
      std::vector<std::shared_ptr<resource::SVMaterial>> const &materials);

  std::shared_ptr<resource::SVMesh>
  createTriangleMesh(std::vector<glm::vec3> const &vertices,
                     std::vector<uint32_t> const &indices,
                     std::vector<glm::vec3> const &normals = {},
                     std::vector<glm::vec2> const &uvs = {});

  struct PhysicalDeviceInfo {
    vk::PhysicalDevice device{};
    bool present{};
    bool supported{};
    int cudaId{-1};
    int pciBus{-1};
    int queueIndex{-1};
  };

private:
  void createInstance();
  void pickSuitableGpuAndQueueFamilyIndex();
  void createDevice();
  void createMemoryAllocator();

  void createCommandPool();
  void createDescriptorPool();

  std::vector<PhysicalDeviceInfo>
  summarizeDeviceInfo(VkSurfaceKHR tmpSurface = nullptr);
};

} // namespace core
} // namespace svulkan2
