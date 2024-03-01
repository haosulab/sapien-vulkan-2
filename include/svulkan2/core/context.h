#pragma once
#include "allocator.h"
#include "command_pool.h"
#include "queue.h"
#include "svulkan2/common/vk.h"
#include "svulkan2/renderer/gui.h"
#include "svulkan2/resource/manager.h"
#include <future>

namespace svulkan2 {

namespace logger {

void setLogLevel(std::string_view level);
std::string getLogLevel();

}; // namespace logger

namespace core {

class Instance;
class PhysicalDevice;
class Device;
class Allocator;

class Context : public std::enable_shared_from_this<Context> {
public:
  static std::shared_ptr<Context> Get();
  static std::shared_ptr<Context> Create(uint32_t maxNumMaterials = 5000,
                                         uint32_t maxNumTextures = 5000,
                                         uint32_t defaultMipLevels = 1,
                                         bool doNotLoadTexture = false, std::string device = "");
  std::shared_ptr<resource::SVResourceManager> createResourceManager();

  ~Context();
  inline bool isVulkanAvailable() const { return mInstance && mPhysicalDevice; }

  bool isPresentAvailable() const;
  bool isRayTracingAvailable() const;

  Queue &getQueue() const;
  Allocator &getAllocator();

  std::unique_ptr<CommandPool> createCommandPool() const;

  uint32_t getGraphicsQueueFamilyIndex() const;

  inline std::shared_ptr<Device> getDevice2() const { return mDevice; };

  vk::Instance getInstance() const;
  vk::Device getDevice() const;

  vk::PhysicalDevice getPhysicalDevice() const;
  std::shared_ptr<PhysicalDevice> getPhysicalDevice2() const;
  vk::PhysicalDeviceLimits const &getPhysicalDeviceLimits() const;

  inline std::mutex &getGlobalLock() { return mGlobalLock; }

  inline DynamicDescriptorPool &getDescriptorPool() const { return *mDescriptorPool; }

  inline vk::DescriptorSetLayout getMetallicDescriptorSetLayout() const {
    return mMetallicDescriptorSetLayout.get();
  }

  std::shared_ptr<resource::SVResourceManager> getResourceManager() const;

  std::unique_ptr<renderer::GuiWindow> createWindow(uint32_t width, uint32_t height);

  std::shared_ptr<resource::SVMesh> createTriangleMesh(std::vector<glm::vec3> const &vertices,
                                                       std::vector<uint32_t> const &indices,
                                                       std::vector<glm::vec3> const &normals = {},
                                                       std::vector<glm::vec2> const &uvs = {});

  inline bool shouldNotLoadTexture() const { return mDoNotLoadTexture; }

  vk::UniqueSemaphore createTimelineSemaphore(uint64_t initialValue);

  vk::Sampler createSampler(vk::SamplerCreateInfo const &info);

private:
  std::shared_ptr<Instance> mInstance;
  std::shared_ptr<PhysicalDevice> mPhysicalDevice;
  std::shared_ptr<Device> mDevice;

  std::mutex mGlobalLock{};
  std::unique_ptr<DynamicDescriptorPool> mDescriptorPool;

  uint32_t mMaxNumMaterials;
  uint32_t mMaxNumTextures;
  uint32_t mDefaultMipLevels;
  bool mDoNotLoadTexture;

  std::weak_ptr<resource::SVResourceManager> mResourceManager;

  vk::UniqueDescriptorSetLayout mMetallicDescriptorSetLayout;

  Context(uint32_t maxNumMaterials = 5000, uint32_t maxNumTextures = 5000,
          uint32_t defaultMipLevels = 1, bool doNotLoadTexture = false, std::string device = "");

  void createDescriptorPool();

  std::mutex mSamplerLock{};
  std::map<vk::SamplerCreateInfo, vk::UniqueSampler> mSamplerRegistry;
};

} // namespace core
} // namespace svulkan2
