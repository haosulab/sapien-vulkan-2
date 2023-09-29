#include "svulkan2/core/context.h"
#include "../common/cuda_helper.h"
#include "../common/logger.h"
#include "svulkan2/common/launch_policy.h"
#include "svulkan2/core/allocator.h"
#include "svulkan2/core/device.h"
#include "svulkan2/core/instance.h"
#include "svulkan2/core/physical_device.h"
#include "svulkan2/shader/glsl_compiler.h"
#include <GLFW/glfw3.h>
#include <easy/profiler.h>
#include <iomanip>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace svulkan2 {
namespace core {

static std::weak_ptr<Context> gContext{};
std::shared_ptr<Context> Context::Get() {
  auto context = gContext.lock();
  if (!context) {
    throw std::runtime_error("Renderer is not created. Renderer creation is "
                             "required before any other operation.");
  }
  return context;
}

std::shared_ptr<Context> Context::Create(bool present, uint32_t maxNumMaterials,
                                         uint32_t maxNumTextures, uint32_t defaultMipLevels,
                                         bool doNotLoadTexture, std::string device) {
  if (auto context = gContext.lock()) {
    if (context->mDefaultMipLevels != defaultMipLevels &&
        context->mDoNotLoadTexture != doNotLoadTexture) {
      logger::warn("Creating multiple renderers with different parameters is not allowed!");
    }
    context->mDefaultMipLevels = defaultMipLevels;
    context->mDoNotLoadTexture = doNotLoadTexture;
    return context;
  }
  auto context = std::shared_ptr<Context>(new Context(present, maxNumMaterials, maxNumTextures,
                                                      defaultMipLevels, doNotLoadTexture, device));
  gContext = context;
  return context;
}

Context::Context(bool present, uint32_t maxNumMaterials, uint32_t maxNumTextures,
                 uint32_t defaultMipLevels, bool doNotLoadTexture, std::string device)
    : mMaxNumMaterials(maxNumMaterials), mMaxNumTextures(maxNumTextures),
      mDefaultMipLevels(defaultMipLevels), mDoNotLoadTexture(doNotLoadTexture) {

#ifdef SVULKAN2_PROFILE
  profiler::startListen();
#endif

  mInstance = Instance::Create(present, VK_MAKE_VERSION(0, 0, 1), VK_MAKE_VERSION(0, 0, 1),
                               VK_API_VERSION_1_2);
  if (!mInstance) {
    return;
  }

  mPhysicalDevice = mInstance->createPhysicalDevice(device);
  if (!mPhysicalDevice) {
    return;
  }

  mDevice = mPhysicalDevice->createDevice();

  createDescriptorPool();

  GLSLCompiler::InitializeProcess();
}

Context::~Context() {
  GLSLCompiler::FinalizeProcess();

  if (mDevice) {
    getDevice().waitIdle();
  }

  logger::info("Vulkan finished");
}

std::shared_ptr<resource::SVResourceManager> Context::getResourceManager() const {
  if (mResourceManager.expired()) {
    throw std::runtime_error(
        "failed to get resource manager: it is not created or already destroyed.");
  }
  return mResourceManager.lock();
}

std::shared_ptr<resource::SVResourceManager> Context::createResourceManager() {
  auto manager = std::make_shared<resource::SVResourceManager>();
  manager->setDefaultMipLevels(mDefaultMipLevels);
  mResourceManager = manager;
  return manager;
}

void Context::createDescriptorPool() {
  if (!isVulkanAvailable()) {
    return;
  }

  mDescriptorPool = std::make_unique<DynamicDescriptorPool>(
      mDevice,
      std::vector{
          vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, mMaxNumTextures},
          vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, mMaxNumMaterials},
          vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 10}});

  {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.push_back(vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                                      vk::ShaderStageFlagBits::eFragment));
    for (uint32_t i = 0; i < 6; ++i) {
      // 6 textures
      bindings.push_back(vk::DescriptorSetLayoutBinding(i + 1,
                                                        vk::DescriptorType::eCombinedImageSampler,
                                                        1, vk::ShaderStageFlagBits::eFragment));
    }
    mMetallicDescriptorSetLayout = getDevice().createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo({}, bindings.size(), bindings.data()));
  }
}

std::unique_ptr<CommandPool> Context::createCommandPool() const {
  EASY_FUNCTION();
  return std::make_unique<CommandPool>();
}

vk::UniqueSemaphore Context::createTimelineSemaphore(uint64_t initialValue = 0) {
  vk::SemaphoreTypeCreateInfo timelineCreateInfo(vk::SemaphoreType::eTimeline, initialValue);
  vk::SemaphoreCreateInfo createInfo{};
  createInfo.setPNext(&timelineCreateInfo);
  return getDevice().createSemaphoreUnique(createInfo);
}

vk::Sampler Context::createSampler(vk::SamplerCreateInfo const &info) {
  std::lock_guard<std::mutex> lock(mSamplerLock);
  auto it = mSamplerRegistry.find(info);
  if (it != mSamplerRegistry.end()) {
    return it->second.get();
  }
  auto samplerUnique = core::Context::Get()->getDevice().createSamplerUnique(info);
  auto sampler = samplerUnique.get();
  mSamplerRegistry[info] = std::move(samplerUnique);
  return sampler;
}

std::unique_ptr<renderer::GuiWindow> Context::createWindow(uint32_t width, uint32_t height) {
  if (!isVulkanAvailable()) {
    throw std::runtime_error("Vulkan is not initialized");
  }

  if (!isPresentAvailable()) {
    throw std::runtime_error("Create window failed: Renderer does not support display.");
  }
  if (width == 0 || height == 0) {
    throw std::runtime_error("Create window failed: width and height must be positive.");
  }
  return std::make_unique<renderer::GuiWindow>(
      std::vector<vk::Format>{vk::Format::eB8G8R8A8Unorm, vk::Format::eR8G8B8A8Unorm,
                              vk::Format::eB8G8R8Unorm, vk::Format::eR8G8B8Unorm},
      vk::ColorSpaceKHR::eSrgbNonlinear, width, height,
      std::vector<vk::PresentModeKHR>{vk::PresentModeKHR::eMailbox}, 2);
};

std::shared_ptr<resource::SVMesh> Context::createTriangleMesh(
    std::vector<glm::vec3> const &vertices, std::vector<uint32_t> const &indices,
    std::vector<glm::vec3> const &normals, std::vector<glm::vec2> const &uvs) {
  auto mesh = std::make_shared<resource::SVMeshRigid>();

  std::vector<float> vertices_;
  for (auto &v : vertices) {
    vertices_.push_back(v.x);
    vertices_.push_back(v.y);
    vertices_.push_back(v.z);
  }
  mesh->setVertexAttribute("position", vertices_);

  if (normals.size()) {
    std::vector<float> normals_;
    for (auto &v : normals) {
      normals_.push_back(v.x);
      normals_.push_back(v.y);
      normals_.push_back(v.z);
    }
    mesh->setVertexAttribute("normal", normals_);
  }

  if (uvs.size()) {
    std::vector<float> uvs_;
    for (auto &v : uvs) {
      uvs_.push_back(v.x);
      uvs_.push_back(v.y);
    }
    mesh->setVertexAttribute("uv", uvs_);
  }

  return mesh;
}

vk::Instance Context::getInstance() const {
  if (!mInstance) {
    return {};
  }
  return mInstance->getInternal();
}
bool Context::isPresentAvailable() const {
  if (!mInstance || !mPhysicalDevice) {
    return false;
  }
  return mInstance->isGLFWEnabled() && mPhysicalDevice->getPickedDeviceInfo().present;
}
bool Context::isRayTracingAvailable() const {
  if (!mPhysicalDevice) {
    return false;
  }
  return mPhysicalDevice->getPickedDeviceInfo().rayTracing;
}
uint32_t Context::getGraphicsQueueFamilyIndex() const {
  if (!mPhysicalDevice) {
    return -1;
  }
  return mPhysicalDevice->getPickedDeviceInfo().queueIndex;
}
vk::PhysicalDevice Context::getPhysicalDevice() const {
  if (!mPhysicalDevice) {
    return {};
  }
  return mPhysicalDevice->getPickedDeviceInfo().device;
}
vk::PhysicalDeviceLimits const &Context::getPhysicalDeviceLimits() const {
  if (!mPhysicalDevice) {
    throw std::runtime_error("invalid physical device");
  }
  return mPhysicalDevice->getPickedDeviceLimits();
}
vk::Device Context::getDevice() const { return mDevice->getInternal(); }
Queue &Context::getQueue() const { return mDevice->getQueue(); }
Allocator &Context::getAllocator() { return mDevice->getAllocator(); }

} // namespace core
} // namespace svulkan2
