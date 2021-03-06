#include "svulkan2/core/context.h"
#include "svulkan2/common/log.h"
#include <GLFW/glfw3.h>

#include "svulkan2/core/allocator.h"

namespace svulkan2 {
namespace core {

static void glfwErrorCallback(int error_code, const char *description) {
  log::error("GLFW error: {}", description);
}

#ifdef VK_VALIDATION
static bool checkValidationLayerSupport() {
  static const std::vector<const char *> validationLayers = {
      "VK_LAYER_KHRONOS_validation"};
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
  for (const char *layerName : validationLayers) {
    bool layerFound = false;

    for (const auto &layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }
    if (!layerFound) {
      return false;
    }
  }
  return true;
}
#endif

Context::Context(uint32_t apiVersion, bool present, uint32_t maxNumMaterials,
                 uint32_t maxNumTextures, uint32_t defaultMipLevels)
    : mApiVersion(apiVersion), mPresent(present),
      mMaxNumMaterials(maxNumMaterials), mMaxNumTextures(maxNumTextures) {
  createInstance();
  pickSuitableGpuAndQueueFamilyIndex();
  createDevice();
  createMemoryAllocator();
  createCommandPool();
  createDescriptorPool();
  mResourceManager = std::make_unique<resource::SVResourceManager>();
  mResourceManager->setDefaultMipLevels(defaultMipLevels);
}

Context::~Context() {}

void Context::createInstance() {
  if (mPresent) {
    log::info("Initializing GLFW");
    glfwInit();
    glfwSetErrorCallback(glfwErrorCallback);
  }

#ifdef VK_VALIDATION
  if (!checkValidationLayerSupport()) {
    throw std::runtime_error(
        "createInstance: validation layers are not available");
  }
  std::vector<const char *> enabledLayers = {"VK_LAYER_KHRONOS_validation"};
#else
  std::vector<const char *> enabledLayers = {};
#endif
  vk::ApplicationInfo appInfo("Vulkan Renderer", VK_MAKE_VERSION(0, 0, 1),
                              "No Engine", VK_MAKE_VERSION(0, 0, 1),
                              mApiVersion);

  std::vector<const char *> instanceExtensions;

#ifdef CUDA_INTEROP
  instanceExtensions.push_back(
      VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  instanceExtensions.push_back(
      VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
#endif

  if (mPresent) {
    if (!glfwVulkanSupported()) {
      log::error("createInstance: present requested but GLFW does not support "
                 "Vulkan. Continue without GLFW.");
      mPresent = false;
    } else {
      uint32_t glfwExtensionCount = 0;
      const char **glfwExtensions =
          glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
      if (!glfwExtensions) {
        throw std::runtime_error(
            "createInstance: Vulkan does not support GLFW extensions");
      }
      for (uint32_t i = 0; i < glfwExtensionCount; ++i) {
        instanceExtensions.push_back(glfwExtensions[i]);
      }
    }
  }
  vk::InstanceCreateInfo createInfo(
      {}, &appInfo, enabledLayers.size(), enabledLayers.data(),
      instanceExtensions.size(), instanceExtensions.data());
  mInstance = vk::createInstanceUnique(createInfo);
}

void Context::pickSuitableGpuAndQueueFamilyIndex() {
  GLFWwindow *window;
  VkSurfaceKHR tmpSurface;
  if (mPresent) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(1, 1, "vulkan", nullptr, nullptr);
    auto result =
        glfwCreateWindowSurface(mInstance.get(), window, nullptr, &tmpSurface);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Window creation failed, you may not create "
                               "GLFW window for presenting rendered results.");
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
  }

  vk::PhysicalDevice pickedDevice;
  uint32_t pickedIndex;
  for (auto device : mInstance->enumeratePhysicalDevices()) {
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
        device.getQueueFamilyProperties();
    pickedIndex = queueFamilyProperties.size();
    for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
      if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
        if (mPresent && !device.getSurfaceSupportKHR(i, tmpSurface)) {
          continue;
        }
        pickedIndex = i;
      }
    }
    if (pickedIndex == queueFamilyProperties.size()) {
      continue;
    }
    auto features = device.getFeatures();
    if (!features.independentBlend) {
      continue;
    }
    pickedDevice = device;
    break;
  }
  if (!pickedDevice) {
    throw std::runtime_error(
        "pickSuitableGpuAndQueue: no compatible GPU found");
  }
  if (mPresent) {
    vkDestroySurfaceKHR(mInstance.get(), tmpSurface, nullptr);
    glfwDestroyWindow(window);
  }
  mPhysicalDevice = pickedDevice;
  mQueueFamilyIndex = pickedIndex;
}

void Context::createDevice() {
  float queuePriority = 0.0f;
  vk::DeviceQueueCreateInfo deviceQueueCreateInfo({}, mQueueFamilyIndex, 1,
                                                  &queuePriority);
  std::vector<const char *> deviceExtensions{};
  vk::PhysicalDeviceFeatures features;
  features.independentBlend = true;
  features.imageCubeArray = true;

#ifdef CUDA_INTEROP
  deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
  deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
  deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
  deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif

  if (mPresent) {
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }
  mDevice = mPhysicalDevice.createDeviceUnique(vk::DeviceCreateInfo(
      {}, 1, &deviceQueueCreateInfo, 0, nullptr, deviceExtensions.size(),
      deviceExtensions.data(), &features));
}

void Context::createMemoryAllocator() {
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.vulkanApiVersion = mApiVersion;
  allocatorInfo.physicalDevice = mPhysicalDevice;
  allocatorInfo.device = mDevice.get();
  allocatorInfo.instance = mInstance.get();
  mAllocator = std::make_unique<Allocator>(*this, allocatorInfo);
}

void Context::createCommandPool() {
  mCommandPool = mDevice->createCommandPoolUnique(vk::CommandPoolCreateInfo(
      vk::CommandPoolCreateFlagBits::eResetCommandBuffer, mQueueFamilyIndex));
}

void Context::createDescriptorPool() {
  vk::DescriptorPoolSize pool_sizes[] = {
      {vk::DescriptorType::eCombinedImageSampler, mMaxNumTextures},
      {vk::DescriptorType::eUniformBuffer, mMaxNumMaterials}};
  auto info = vk::DescriptorPoolCreateInfo(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      mMaxNumTextures + mMaxNumMaterials, 2, pool_sizes);
  mDescriptorPool = getDevice().createDescriptorPoolUnique(info);

  {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.push_back(
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                       vk::ShaderStageFlagBits::eFragment));
    for (uint32_t i = 0; i < 4; ++i) {
      bindings.push_back(vk::DescriptorSetLayoutBinding(
          i + 1, vk::DescriptorType::eCombinedImageSampler, 1,
          vk::ShaderStageFlagBits::eFragment));
    }
    mMetallicDescriptorSetLayout = mDevice->createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo({}, bindings.size(),
                                          bindings.data()));
  }
  {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.push_back(
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                       vk::ShaderStageFlagBits::eFragment));
    for (uint32_t i = 0; i < 3; ++i) {
      bindings.push_back(vk::DescriptorSetLayoutBinding(
          i + 1, vk::DescriptorType::eCombinedImageSampler, 1,
          vk::ShaderStageFlagBits::eFragment));
    }
    mSpecularDescriptorSetLayout = mDevice->createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo({}, bindings.size(),
                                          bindings.data()));
  }
}

vk::UniqueCommandBuffer
Context::createCommandBuffer(vk::CommandBufferLevel level) const {
  return std::move(
      mDevice->allocateCommandBuffersUnique({mCommandPool.get(), level, 1})
          .front());
}

void Context::submitCommandBufferAndWait(
    vk::CommandBuffer commandBuffer) const {
  auto fence = mDevice->createFenceUnique({});
  getQueue().submit(vk::SubmitInfo(0, nullptr, nullptr, 1, &commandBuffer),
                    fence.get());
  mDevice->waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
}

vk::UniqueFence
Context::submitCommandBufferForFence(vk::CommandBuffer commandBuffer) const {
  auto fence = mDevice->createFenceUnique({});
  getQueue().submit(vk::SubmitInfo(0, nullptr, nullptr, 1, &commandBuffer),
                    fence.get());
  return fence;
}

std::future<void>
Context::submitCommandBuffer(vk::CommandBuffer commandBuffer) const {
  auto fence = mDevice->createFenceUnique({});
  getQueue().submit(vk::SubmitInfo(0, nullptr, nullptr, 1, &commandBuffer),
                    fence.get());
  return std::async(std::launch::async, [fence = std::move(fence), this]() {
    mDevice->waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
  });
}

vk::Queue Context::getQueue() const {
  return mDevice->getQueue(mQueueFamilyIndex, 0);
}

std::unique_ptr<renderer::GuiWindow> Context::createWindow(uint32_t width,
                                                           uint32_t height) {
  if (!mPresent) {
    throw std::runtime_error(
        "Create window failed: context is not created with present support");
  }
  if (width == 0 || height == 0) {
    throw std::runtime_error(
        "Create window failed: width and height must be positive.");
  }
  return std::make_unique<renderer::GuiWindow>(
      *this,
      std::vector<vk::Format>{
          vk::Format::eB8G8R8A8Unorm, vk::Format::eR8G8B8A8Unorm,
          vk::Format::eB8G8R8Unorm, vk::Format::eR8G8B8Unorm},
      vk::ColorSpaceKHR::eSrgbNonlinear, width, height,
      std::vector<vk::PresentModeKHR>{vk::PresentModeKHR::eFifo}, 2);
};

std::shared_ptr<resource::SVMetallicMaterial>
Context::createMetallicMaterial(glm::vec4 baseColor, float fresnel,
                                float roughness, float metallic,
                                float transparency) {
  return std::make_shared<resource::SVMetallicMaterial>(
      baseColor, fresnel, roughness, metallic, transparency);
}

std::shared_ptr<resource::SVSpecularMaterial>
Context::createSpecularMaterial(glm::vec4 diffuse, glm::vec4 specular,
                                float transparency) {
  return std::make_shared<resource::SVSpecularMaterial>(diffuse, specular,
                                                        transparency);
}

std::shared_ptr<resource::SVModel> Context::createModel(
    std::vector<std::shared_ptr<resource::SVMesh>> meshes,
    std::vector<std::shared_ptr<resource::SVMaterial>> materials) {
  if (meshes.size() != materials.size()) {
    throw std::runtime_error(
        "create model failed: meshes and materials must have the same size.");
  }
  std::vector<std::shared_ptr<resource::SVShape>> shapes;
  for (uint32_t i = 0; i < meshes.size(); ++i) {
    auto shape = std::make_shared<resource::SVShape>();
    shape->mesh = meshes[i];
    shape->material = materials[i];
    shapes.push_back(shape);
  }
  return resource::SVModel::FromData(shapes);
}

std::shared_ptr<resource::SVMesh>
Context::createTriangleMesh(std::vector<glm::vec3> const &vertices,
                            std::vector<uint32_t> const &indices,
                            std::vector<glm::vec3> const &normals,
                            std::vector<glm::vec2> const &uvs) {
  auto mesh = std::make_shared<resource::SVMesh>();

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

} // namespace core
} // namespace svulkan2
