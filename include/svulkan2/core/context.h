#pragma once
#include "allocator.h"
#include "command_pool.h"
#include "queue.h"
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
  vk::PhysicalDeviceLimits mPhysicalDeviceLimits;
  uint32_t mQueueFamilyIndex;

  vk::UniqueDevice mDevice;
  std::unique_ptr<class Allocator> mAllocator;
  std::unique_ptr<Queue> mQueue;

  std::mutex mGlobalLock{};
  vk::UniqueDescriptorPool mDescriptorPool;

  uint32_t mMaxNumMaterials;
  uint32_t mMaxNumTextures;
  uint32_t mDefaultMipLevels;
  bool mDoNotLoadTexture;

  std::string mDeviceHint;

  std::weak_ptr<resource::SVResourceManager> mResourceManager;

  vk::UniqueDescriptorSetLayout mMetallicDescriptorSetLayout;

  Context(bool present = true, uint32_t maxNumMaterials = 5000,
          uint32_t maxNumTextures = 5000, uint32_t defaultMipLevels = 1,
          bool doNotLoadTexture = false, std::string device = "");

  void init();

public:
  static std::shared_ptr<Context> Get();
  static std::shared_ptr<Context>
  Create(bool present = true, uint32_t maxNumMaterials = 5000,
         uint32_t maxNumTextures = 5000, uint32_t defaultMipLevels = 1,
         bool doNotLoadTexture = false, std::string device = "");
  std::shared_ptr<resource::SVResourceManager> createResourceManager();

  ~Context();

  inline bool isVulkanAvailable() const { return mVulkanAvailable; }
  inline bool isPresentAvailable() const { return mPresent; }

  inline Queue &getQueue() const { return *mQueue.get(); }
  inline class Allocator &getAllocator() { return *mAllocator; }

  std::unique_ptr<CommandPool> createCommandPool() const;

  inline uint32_t getGraphicsQueueFamilyIndex() const {
    return mQueueFamilyIndex;
  }
  inline vk::Instance getInstance() const { return mInstance.get(); }
  inline vk::Device getDevice() const { return mDevice.get(); }
  inline vk::PhysicalDevice getPhysicalDevice() const {
    return mPhysicalDevice;
  }
  inline vk::PhysicalDeviceLimits const &getPhysicalDeviceLimits() const {
    return mPhysicalDeviceLimits;
  }

  inline std::mutex &getGlobalLock() { return mGlobalLock; }
  inline vk::DescriptorPool getDescriptorPool() const {
    return mDescriptorPool.get();
  }
  inline vk::DescriptorSetLayout getMetallicDescriptorSetLayout() const {
    return mMetallicDescriptorSetLayout.get();
  }

  std::shared_ptr<resource::SVResourceManager> getResourceManager() const;

  std::unique_ptr<renderer::GuiWindow> createWindow(uint32_t width,
                                                    uint32_t height);

  std::shared_ptr<resource::SVMetallicMaterial>
  createMetallicMaterial(glm::vec4 emission, glm::vec4 baseColor, float fresnel,
                         float roughness, float metallic, float transparency);

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

  inline bool shouldNotLoadTexture() const { return mDoNotLoadTexture; }

  vk::UniqueSemaphore createTimelineSemaphore(uint64_t initialValue);

private:
  void createInstance();
  void pickSuitableGpuAndQueueFamilyIndex();
  void createDevice();
  void createMemoryAllocator();

  void createDescriptorPool();

  std::vector<PhysicalDeviceInfo>
  summarizeDeviceInfo(VkSurfaceKHR tmpSurface = nullptr);
};

} // namespace core
} // namespace svulkan2
