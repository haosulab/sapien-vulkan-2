#include "svulkan2/core/context.h"
#include "svulkan2/common/cuda_helper.h"
#include "svulkan2/common/launch_policy.h"
#include "svulkan2/common/log.h"
#include "svulkan2/core/allocator.h"
#include <GLFW/glfw3.h>
#include <easy/profiler.h>
#include <iomanip>

#if !defined(VULKAN_HPP_STORAGE_SHARED)
#define VULKAN_HPP_STORAGE_SHARED
#define VULKAN_HPP_STORAGE_SHARED_EXPORT
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

namespace svulkan2 {
namespace core {

static std::weak_ptr<Context> gInstance{};
std::shared_ptr<Context> Context::Get() {
  auto instance = gInstance.lock();
  if (!instance) {
    throw std::runtime_error("Renderer is not created. Renderer creation is "
                             "required before any other operation.");
  }
  return instance;
}

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

std::shared_ptr<Context> Context::Create(bool present, uint32_t maxNumMaterials,
                                         uint32_t maxNumTextures,
                                         uint32_t defaultMipLevels,
                                         bool doNotLoadTexture,
                                         std::string device) {
  if (!gInstance.expired()) {
    log::warn("Only 1 renderer is allowed per process. All previously created "
              "renderer resources are now invalid");
  }
  auto context = std::shared_ptr<Context>(
      new Context(present, maxNumMaterials, maxNumTextures, defaultMipLevels,
                  doNotLoadTexture, device));
  gInstance = context;
  context->init();
  return context;
}

Context::Context(bool present, uint32_t maxNumMaterials,
                 uint32_t maxNumTextures, uint32_t defaultMipLevels,
                 bool doNotLoadTexture, std::string device)
    : mApiVersion(VK_API_VERSION_1_2), mPresent(present),
      mMaxNumMaterials(maxNumMaterials), mMaxNumTextures(maxNumTextures),
      mDefaultMipLevels(defaultMipLevels), mDoNotLoadTexture(doNotLoadTexture),
      mDeviceHint(device) {}

void Context::init() {
  profiler::startListen();
  createInstance();
  pickSuitableGpuAndQueueFamilyIndex();
  createDevice();
  createMemoryAllocator();
  createDescriptorPool();
}

Context::~Context() {
  if (mDevice) {
    mDevice->waitIdle();
  }
  if (mPresent) {
    glfwTerminate();
    log::info("GLFW terminated");
  }

  log::info("Vulkan finished");
}

std::shared_ptr<resource::SVResourceManager>
Context::getResourceManager() const {
  if (mResourceManager.expired()) {
    throw std::runtime_error(
        "failed to get resource manager: destroyed or not created");
  }
  return mResourceManager.lock();
}

std::shared_ptr<resource::SVResourceManager> Context::createResourceManager() {
  auto manager = std::make_shared<resource::SVResourceManager>();
  manager->setDefaultMipLevels(mDefaultMipLevels);
  mResourceManager = manager;
  return manager;
}

void Context::createInstance() {
  if (mPresent) {
    glfwSetErrorCallback(glfwErrorCallback);
    if (glfwInit()) {
      log::info("GLFW initialized.");
    } else {
      log::warn("Continue without GLFW.");
      mPresent = false;
    }
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
        int glfwExtensionsErrCode = glfwGetError(NULL);
        if (glfwExtensionsErrCode == GLFW_NOT_INITIALIZED) {
          throw std::runtime_error("createInstance: GLFW has not initialized");
        } else if (glfwExtensionsErrCode == GLFW_API_UNAVAILABLE) {
          throw std::runtime_error(
              "createInstance: Vulkan is not available on the machine");
        } else
          throw std::runtime_error(
              "createInstance: No Vulkan extensions found for window "
              "surface creation (hint: set VK_ICD_FILENAMES to `locate "
              "icd.json`).");
      }
      for (uint32_t i = 0; i < glfwExtensionCount; ++i) {
        instanceExtensions.push_back(glfwExtensions[i]);
      }
    }
  }
  vk::DynamicLoader dl;
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
      dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

  vk::InstanceCreateInfo createInfo(
      {}, &appInfo, enabledLayers.size(), enabledLayers.data(),
      instanceExtensions.size(), instanceExtensions.data());
  mVulkanAvailable = false;
  try {
    mInstance = vk::createInstanceUnique(createInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(mInstance.get());
    mVulkanAvailable = true;
    log::info("Vulkan instance initialized");
  } catch (vk::OutOfHostMemoryError const &err) {
    throw err;
  } catch (vk::OutOfDeviceMemoryError const &err) {
    throw err;
  } catch (vk::InitializationFailedError const &err) {
    log::error("Vulkan initialization failed. You may not use the renderer to "
               "render, however, CPU resources will be still available.");
  } catch (vk::LayerNotPresentError const &err) {
    log::error(
        "Some required Vulkan layer is not present. You may not use the "
        "renderer to render, however, CPU resources will be still available.");
  } catch (vk::ExtensionNotPresentError const &err) {
    log::error(
        "Some required Vulkan extension is not present. You may not use the "
        "renderer to render, however, CPU resources will be still available.");
  } catch (vk::IncompatibleDriverError const &err) {
    log::error(
        "Vulkan is incompatible with your driver. You may not use the renderer "
        "to render, however, CPU resources will be still available.");
  }
}

std::vector<Context::PhysicalDeviceInfo>
Context::summarizeDeviceInfo(VkSurfaceKHR tmpSurface) {
  std::vector<Context::PhysicalDeviceInfo> devices;

  std::stringstream ss;
  ss << "Devices visible to Vulkan" << std::endl;
  ss << std::setw(3) << "" << std::setw(20) << "name" << std::setw(10)
     << "Present" << std::setw(10) << "Supported" << std::setw(10) << "PciBus"
     << std::setw(10) << "CudaId" << std::endl;

  vk::PhysicalDeviceFeatures2 features{};
  vk::PhysicalDeviceDescriptorIndexingFeatures descriptorFeatures{};
  vk::PhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures{};
  features.setPNext(&descriptorFeatures);
  descriptorFeatures.setPNext(&timelineSemaphoreFeatures);

  int ord = 0;
  for (auto device : mInstance->enumeratePhysicalDevices()) {
    std::string name;
    bool present = false;
    bool required_features = false;
    int cudaId = -1;
    int busid = -1;
    int queueIdxNoPresent = -1;
    int queueIdxPresent = -1;
    int queueIdx = -1;

    std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
        device.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
      if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics &&
          queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute) {
        if (queueIdxNoPresent == -1) {
          queueIdxNoPresent = i;
        }
        if (mPresent && device.getSurfaceSupportKHR(i, tmpSurface)) {
          queueIdxPresent = i;
          present = true;
          break;
        }
      }
    }

    if (queueIdxPresent != -1) {
      queueIdx = queueIdxPresent;
    } else {
      queueIdx = queueIdxNoPresent;
    }

    device.getFeatures2(&features);
    if (features.features.independentBlend && features.features.wideLines &&
        features.features.geometryShader &&
        descriptorFeatures.descriptorBindingPartiallyBound &&
        timelineSemaphoreFeatures.timelineSemaphore) {
      required_features = true;
    }

    auto properties = device.getProperties();
    name = std::string(properties.deviceName);

    vk::PhysicalDeviceProperties2KHR p2;
    vk::PhysicalDevicePCIBusInfoPropertiesEXT pciInfo;
    pciInfo.pNext = p2.pNext;
    p2.pNext = &pciInfo;
    device.getProperties2(&p2);
    busid = pciInfo.pciBus;

#ifdef CUDA_INTEROP
    cudaId = getCudaDeviceIdFromPhysicalDevice(device);
#endif

    bool supported = required_features && queueIdx != -1;

    ss << std::setw(3) << ord++ << std::setw(20) << name.substr(0, 20)
       << std::setw(10) << present << std::setw(10) << supported << std::hex
       << std::setw(10) << busid << std::dec << std::setw(10)
       << (cudaId == -1 ? "No Device" : std::to_string(cudaId)) << std::endl;

    devices.push_back(
        Context::PhysicalDeviceInfo{.device = device,
                                    .present = present,
                                    .supported = required_features,
                                    .cudaId = cudaId,
                                    .pciBus = busid,
                                    .queueIndex = queueIdx});
  }
  log::info(ss.str());

#ifdef CUDA_INTEROP
  ss = {};
  ss << "Devices visible to Cuda" << std::endl;
  ss << std::setw(10) << "CudaId" << std::setw(10) << "PciBus" << std::setw(25)
     << "PciBusString" << std::endl;
  for (uint32_t i = 0; i < 20; ++i) {
    int busId = getPCIBusIdFromCudaDeviceId(i);
    std::string pciBus(20, '\0');
    auto result = cudaDeviceGetPCIBusId(pciBus.data(), 20, i);

    switch (result) {
    case cudaErrorInvalidValue:
      pciBus = "Invalid Value";
      break;
    case cudaErrorInvalidDevice:
      pciBus = "Invalid Device";
      break;
    case cudaErrorInitializationError:
      pciBus = "Not Initialized";
      break;
    case cudaErrorInsufficientDriver:
      pciBus = "Insufficient Driver";
      break;
    case cudaErrorNoDevice:
      pciBus = "No Device";
      break;
    default:
      break;
    }

    ss << std::setw(10) << i << std::hex << std::setw(10) << busId << std::dec
       << std::setw(25) << pciBus.c_str() << std::endl;
  }
  log::info(ss.str());
#endif

  return devices;
}

int pickCudaDevice(std::vector<Context::PhysicalDeviceInfo> devices, int id) {
  int idx = 0;
  for (auto &info : devices) {
    if (info.cudaId == id && info.supported) {
      return idx;
    }
    idx++;
  }
  return -1;
}

int pickCudaDevice(std::vector<Context::PhysicalDeviceInfo> devices) {
  int idx = 0;
  for (auto &info : devices) {
    if (info.cudaId >= 0 && info.supported) {
      return idx;
    }
    idx++;
  }
  return -1;
}

int pickCudaDeviceWithPresent(
    std::vector<Context::PhysicalDeviceInfo> devices) {
  int idx = 0;
  for (auto &info : devices) {
    if (info.cudaId >= 0 && info.present && info.supported) {
      return idx;
    }
    idx++;
  }
  return -1;
}

int pickPciWithPresent(std::vector<Context::PhysicalDeviceInfo> devices,
                       int pci) {
  int idx = 0;
  for (auto &info : devices) {
    if (info.pciBus == pci && info.present && info.supported) {
      return idx;
    }
    idx++;
  }
  return -1;
}

int pickPci(std::vector<Context::PhysicalDeviceInfo> devices, int pci) {
  int idx = 0;
  for (auto &info : devices) {
    if (info.pciBus == pci && info.supported) {
      return idx;
    }
    idx++;
  }
  return -1;
}

int pickPresent(std::vector<Context::PhysicalDeviceInfo> devices) {
  int idx = 0;
  for (auto &info : devices) {
    if (info.present && info.supported) {
      return idx;
    }
    idx++;
  }
  return -1;
}

int pickAny(std::vector<Context::PhysicalDeviceInfo> devices) {
  int idx = 0;
  for (auto &info : devices) {
    if (info.supported) {
      return idx;
    }
    idx++;
  }
  return -1;
}

void Context::pickSuitableGpuAndQueueFamilyIndex() {
  if (!mVulkanAvailable) {
    return;
  }

  GLFWwindow *window{};
  VkSurfaceKHR tmpSurface = nullptr;
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

  auto devices = summarizeDeviceInfo(tmpSurface);

  if (mPresent) {
    vkDestroySurfaceKHR(mInstance.get(), tmpSurface, nullptr);
    glfwDestroyWindow(window);
  }

  int pickedDeviceIdx = -1;
  if (mDeviceHint.starts_with("cuda:")) {
    try {
      int cudaId = std::stoi(mDeviceHint.substr(5));
      pickedDeviceIdx = pickCudaDevice(devices, cudaId);
      if (pickedDeviceIdx == -1) {
        throw std::runtime_error(
            "Cannot find cuda device suitable for rendering " + mDeviceHint);
      }
    } catch (std::invalid_argument const &e) {
      throw std::runtime_error("Invalid device " + mDeviceHint);
    }
  } else if (mDeviceHint.starts_with("cuda")) {
    if (mPresent) {
      pickedDeviceIdx = pickCudaDeviceWithPresent(devices);
    }
    if (pickedDeviceIdx == -1) {
      pickedDeviceIdx = pickCudaDevice(devices);
    }
    if (pickedDeviceIdx == -1) {
      throw std::runtime_error(
          "Cannot find any cuda device suitable for rendering.");
    }
  } else if (mDeviceHint.starts_with("pci:")) {
    int pci = -1;
    try {
      pci = std::stoi(mDeviceHint.substr(4), 0, 16);
    } catch (std::invalid_argument const &e) {
      throw std::runtime_error("Invalid device " + mDeviceHint);
    }
    if (mPresent) {
      pickedDeviceIdx = pickPciWithPresent(devices, pci);
    }
    if (pickedDeviceIdx == -1) {
      pickedDeviceIdx = pickPci(devices, pci);
    }
    if (pickedDeviceIdx == -1) {
      throw std::runtime_error("Cannot find " + mDeviceHint +
                               " for rendering.");
    }
  } else if (mDeviceHint == "") {
    if (mPresent) {
      pickedDeviceIdx = pickCudaDeviceWithPresent(devices);
      if (pickedDeviceIdx == -1) {
        pickedDeviceIdx = pickPresent(devices);
      }
    }
    if (pickedDeviceIdx == -1) {
      pickedDeviceIdx = pickCudaDevice(devices);
    }
    if (pickedDeviceIdx == -1) {
      pickedDeviceIdx = pickAny(devices);
    }
  }

  if (pickedDeviceIdx == -1) {
    throw std::runtime_error("Cannot find a suitable rendering device");
  }

  if (mPresent && !devices[pickedDeviceIdx].present) {
    log::error("Present requested but the selected device does not support "
               "present. Continue without present.");
    mPresent = false;
  }

  log::info("Vulkan picked device: {}", pickedDeviceIdx);

  vk::PhysicalDevice pickedDevice = devices[pickedDeviceIdx].device;

  mPhysicalDevice = pickedDevice;
  mPhysicalDeviceLimits = mPhysicalDevice.getProperties().limits;
  mQueueFamilyIndex = devices[pickedDeviceIdx].queueIndex;
}

void Context::createDevice() {
  if (!mVulkanAvailable) {
    return;
  }

  float queuePriority = 0.0f;
  vk::DeviceQueueCreateInfo deviceQueueCreateInfo({}, mQueueFamilyIndex, 1,
                                                  &queuePriority);
  std::vector<const char *> deviceExtensions{};

  vk::PhysicalDeviceFeatures2 features{};
  vk::PhysicalDeviceDescriptorIndexingFeatures descriptorFeatures{};
  vk::PhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures{};

  features.setPNext(&descriptorFeatures);
  descriptorFeatures.setPNext(&timelineSemaphoreFeatures);

  features.features.setIndependentBlend(true);
  features.features.setWideLines(true);
  features.features.setGeometryShader(true);
  descriptorFeatures.setDescriptorBindingPartiallyBound(true);
  timelineSemaphoreFeatures.setTimelineSemaphore(true);

#ifdef CUDA_INTEROP
  deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
  deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
  deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
  deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif

  if (mPresent) {
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  vk::DeviceCreateInfo deviceInfo({}, deviceQueueCreateInfo, {},
                                  deviceExtensions);
  deviceInfo.setPNext(&features);
  mDevice = mPhysicalDevice.createDeviceUnique(deviceInfo);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(mDevice.get());

  mQueue = std::make_unique<Queue>();
}

void Context::createMemoryAllocator() {
  if (!mVulkanAvailable) {
    return;
  }

  VmaVulkanFunctions vulkanFunctions{};
  vulkanFunctions.vkGetPhysicalDeviceProperties =
      (PFN_vkGetPhysicalDeviceProperties)mInstance->getProcAddr(
          "vkGetPhysicalDeviceProperties");
  vulkanFunctions.vkGetPhysicalDeviceMemoryProperties =
      (PFN_vkGetPhysicalDeviceMemoryProperties)mInstance->getProcAddr(
          "vkGetPhysicalDeviceMemoryProperties");
  vulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR =
      (PFN_vkGetPhysicalDeviceMemoryProperties2)mInstance->getProcAddr(
          "vkGetPhysicalDeviceMemoryProperties2");
  vulkanFunctions.vkAllocateMemory =
      (PFN_vkAllocateMemory)mDevice->getProcAddr("vkAllocateMemory");
  vulkanFunctions.vkFreeMemory =
      (PFN_vkFreeMemory)mDevice->getProcAddr("vkFreeMemory");
  vulkanFunctions.vkMapMemory =
      (PFN_vkMapMemory)mDevice->getProcAddr("vkMapMemory");
  vulkanFunctions.vkUnmapMemory =
      (PFN_vkUnmapMemory)mDevice->getProcAddr("vkUnmapMemory");
  vulkanFunctions.vkFlushMappedMemoryRanges =
      (PFN_vkFlushMappedMemoryRanges)mDevice->getProcAddr(
          "vkFlushMappedMemoryRanges");
  vulkanFunctions.vkInvalidateMappedMemoryRanges =
      (PFN_vkInvalidateMappedMemoryRanges)mDevice->getProcAddr(
          "vkInvalidateMappedMemoryRanges");
  vulkanFunctions.vkBindBufferMemory =
      (PFN_vkBindBufferMemory)mDevice->getProcAddr("vkBindBufferMemory");
  vulkanFunctions.vkBindImageMemory =
      (PFN_vkBindImageMemory)mDevice->getProcAddr("vkBindImageMemory");
  vulkanFunctions.vkGetBufferMemoryRequirements =
      (PFN_vkGetBufferMemoryRequirements)mDevice->getProcAddr(
          "vkGetBufferMemoryRequirements");
  vulkanFunctions.vkGetImageMemoryRequirements =
      (PFN_vkGetImageMemoryRequirements)mDevice->getProcAddr(
          "vkGetImageMemoryRequirements");
  vulkanFunctions.vkCreateBuffer =
      (PFN_vkCreateBuffer)mDevice->getProcAddr("vkCreateBuffer");
  vulkanFunctions.vkDestroyBuffer =
      (PFN_vkDestroyBuffer)mDevice->getProcAddr("vkDestroyBuffer");
  vulkanFunctions.vkCreateImage =
      (PFN_vkCreateImage)mDevice->getProcAddr("vkCreateImage");
  vulkanFunctions.vkDestroyImage =
      (PFN_vkDestroyImage)mDevice->getProcAddr("vkDestroyImage");
  vulkanFunctions.vkCmdCopyBuffer =
      (PFN_vkCmdCopyBuffer)mDevice->getProcAddr("vkCmdCopyBuffer");
  vulkanFunctions.vkGetBufferMemoryRequirements2KHR =
      (PFN_vkGetBufferMemoryRequirements2KHR)mDevice->getProcAddr(
          "vkGetBufferMemoryRequirements2");
  vulkanFunctions.vkGetImageMemoryRequirements2KHR =
      (PFN_vkGetImageMemoryRequirements2KHR)mDevice->getProcAddr(
          "vkGetImageMemoryRequirements2");
  vulkanFunctions.vkBindBufferMemory2KHR =
      (PFN_vkBindBufferMemory2KHR)mDevice->getProcAddr("vkBindBufferMemory2");
  vulkanFunctions.vkBindImageMemory2KHR =
      (PFN_vkBindImageMemory2KHR)mDevice->getProcAddr("vkBindImageMemory2");

  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.physicalDevice = mPhysicalDevice;
  allocatorInfo.device = mDevice.get();
  allocatorInfo.instance = mInstance.get();
  allocatorInfo.pVulkanFunctions = &vulkanFunctions;

  mAllocator = std::make_unique<Allocator>(allocatorInfo);
}

void Context::createDescriptorPool() {
  if (!mVulkanAvailable) {
    return;
  }

  vk::DescriptorPoolSize pool_sizes[] = {
      {vk::DescriptorType::eCombinedImageSampler, mMaxNumTextures},
      {vk::DescriptorType::eUniformBuffer, mMaxNumMaterials},
      {vk::DescriptorType::eStorageImage, 10}, // for some compute
  };
  auto info = vk::DescriptorPoolCreateInfo(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      mMaxNumTextures + mMaxNumMaterials, 3, pool_sizes);
  mDescriptorPool = getDevice().createDescriptorPoolUnique(info);

  {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.push_back(
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                       vk::ShaderStageFlagBits::eFragment));
    for (uint32_t i = 0; i < 5; ++i) {
      bindings.push_back(vk::DescriptorSetLayoutBinding(
          i + 1, vk::DescriptorType::eCombinedImageSampler, 1,
          vk::ShaderStageFlagBits::eFragment));
    }
    mMetallicDescriptorSetLayout = mDevice->createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo({}, bindings.size(),
                                          bindings.data()));
  }
}

std::unique_ptr<CommandPool> Context::createCommandPool() const {
  EASY_FUNCTION();
  return std::make_unique<CommandPool>();
}

vk::UniqueSemaphore
Context::createTimelineSemaphore(uint64_t initialValue = 0) {
  vk::SemaphoreTypeCreateInfo timelineCreateInfo(vk::SemaphoreType::eTimeline,
                                                 initialValue);
  vk::SemaphoreCreateInfo createInfo{};
  createInfo.setPNext(&timelineCreateInfo);
  return mDevice->createSemaphoreUnique(createInfo);
}

std::unique_ptr<renderer::GuiWindow> Context::createWindow(uint32_t width,
                                                           uint32_t height) {
  if (!mVulkanAvailable) {
    throw std::runtime_error("Vulkan is not initialized");
  }

  if (!mPresent) {
    throw std::runtime_error(
        "Create window failed: context is not created with present support");
  }
  if (width == 0 || height == 0) {
    throw std::runtime_error(
        "Create window failed: width and height must be positive.");
  }
  return std::make_unique<renderer::GuiWindow>(
      std::vector<vk::Format>{
          vk::Format::eB8G8R8A8Unorm, vk::Format::eR8G8B8A8Unorm,
          vk::Format::eB8G8R8Unorm, vk::Format::eR8G8B8Unorm},
      vk::ColorSpaceKHR::eSrgbNonlinear, width, height,
      std::vector<vk::PresentModeKHR>{vk::PresentModeKHR::eMailbox}, 2);
};

std::shared_ptr<resource::SVMetallicMaterial>
Context::createMetallicMaterial(glm::vec4 emission, glm::vec4 baseColor,
                                float fresnel, float roughness, float metallic,
                                float transparency) {
  return std::make_shared<resource::SVMetallicMaterial>(
      emission, baseColor, fresnel, roughness, metallic, transparency);
}

std::shared_ptr<resource::SVModel> Context::createModel(
    std::vector<std::shared_ptr<resource::SVMesh>> const &meshes,
    std::vector<std::shared_ptr<resource::SVMaterial>> const &materials) {
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
