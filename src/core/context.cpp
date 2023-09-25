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

  // mAllocator = std::make_unique<Allocator>(mInstance->getApiVersion(), mInstance->getInternal(),
  //                                          mPhysicalDevice->getPickedDeviceInfo().device,
  //                                          mDevice->getInternal());

  createDescriptorPool();

  GLSLCompiler::InitializeProcess();
}

// void Context::init() {

// #ifdef SVULKAN2_PROFILE
//   profiler::startListen();
// #endif

//   // create instance
//   // mInstance = Instance::Create(mPresentRequested, VK_MAKE_VERSION(0, 0, 1),
//   //                              VK_MAKE_VERSION(0, 0, 1), VK_API_VERSION_1_2);

//   // pickSuitableGpuAndQueueFamilyIndex();
//   // createDevice();
//   // createMemoryAllocator();
//   // createDescriptorPool();

//   GLSLCompiler::InitializeProcess();
// }

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

// void Context::createInstance() {
//   // try to load system Vulkan
//   try {
//     mDynamicLoader = std::make_unique<vk::DynamicLoader>();
//   } catch (std::runtime_error &) {
//     mDynamicLoader.reset();
//   }

//   // try to load SAPIEN specified Vulkan
//   if (!mDynamicLoader) {
//     auto path = std::getenv("SAPIEN_VULKAN_LIBRARY_PATH");
//     if (path && std::filesystem::is_regular_file(std::filesystem::path(path))) {
//       try {
//         mDynamicLoader = std::make_unique<vk::DynamicLoader>(path);
//       } catch (std::runtime_error &) {
//         mDynamicLoader.reset();
//       }
//     }
//   }

//   // give up
//   if (!mDynamicLoader) {
//     logger::error("Failed to load vulkan library! You may not use the renderer to "
//                   "render, however, CPU resources will be still available.");
//     return;
//   }

//   if (mPresentRequested) {
//     glfwSetErrorCallback(glfwErrorCallback);
//     if (glfwInit()) {
//       logger::info("GLFW initialized.");
//     } else {
//       logger::warn("Continue without GLFW.");
//       mPresentRequested = false;
//     }
//   }

// #ifdef VK_VALIDATION
//   if (!checkValidationLayerSupport(
//           mDynamicLoader->getProcAddress<PFN_vkEnumerateInstanceLayerProperties>(
//               "vkEnumerateInstanceLayerProperties"))) {
//     throw std::runtime_error("createInstance: validation layers are not available");
//   }
//   std::vector<const char *> enabledLayers = {"VK_LAYER_KHRONOS_validation"};
// #else
//   std::vector<const char *> enabledLayers = {};
// #endif
//   vk::ApplicationInfo appInfo("Vulkan Renderer", VK_MAKE_VERSION(0, 0, 1), "No Engine",
//                               VK_MAKE_VERSION(0, 0, 1), mApiVersion);

//   std::vector<const char *> instanceExtensions;

// #ifdef SVULKAN2_CUDA_INTEROP
//   instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
//   instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
// #endif

//   if (mPresentRequested) {
//     if (!glfwVulkanSupported()) {
//       logger::error("createInstance: present requested but GLFW does not support "
//                     "Vulkan. Continue without GLFW.");
//       mPresentRequested = false;
//     } else {
//       uint32_t glfwExtensionCount = 0;
//       const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
//       if (!glfwExtensions) {
//         int glfwExtensionsErrCode = glfwGetError(NULL);
//         if (glfwExtensionsErrCode == GLFW_NOT_INITIALIZED) {
//           throw std::runtime_error("createInstance: GLFW has not initialized");
//         } else if (glfwExtensionsErrCode == GLFW_API_UNAVAILABLE) {
//           throw std::runtime_error("createInstance: Vulkan is not available on the machine");
//         } else
//           throw std::runtime_error("createInstance: No Vulkan extensions found for window "
//                                    "surface creation (hint: set VK_ICD_FILENAMES to `locate "
//                                    "icd.json`).");
//       }
//       for (uint32_t i = 0; i < glfwExtensionCount; ++i) {
//         instanceExtensions.push_back(glfwExtensions[i]);
//       }
//     }
//   }

//   PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
//       mDynamicLoader->getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
//   VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

//   vk::InstanceCreateInfo createInfo({}, &appInfo, enabledLayers.size(), enabledLayers.data(),
//                                     instanceExtensions.size(), instanceExtensions.data());
//   mVulkanAvailable = false;
//   try {
//     mInstance = vk::createInstanceUnique(createInfo);
//     VULKAN_HPP_DEFAULT_DISPATCHER.init(mInstance.get());
//     mVulkanAvailable = true;
//     logger::info("Vulkan instance initialized");
//   } catch (vk::OutOfHostMemoryError const &err) {
//     throw err;
//   } catch (vk::OutOfDeviceMemoryError const &err) {
//     throw err;
//   } catch (vk::InitializationFailedError const &err) {
//     logger::error("Vulkan initialization failed. You may not use the renderer to "
//                   "render, however, CPU resources will be still available.");
//   } catch (vk::LayerNotPresentError const &err) {
//     logger::error("Some required Vulkan layer is not present. You may not use the "
//                   "renderer to render, however, CPU resources will be still available.");
//   } catch (vk::ExtensionNotPresentError const &err) {
//     logger::error("Some required Vulkan extension is not present. You may not use the "
//                   "renderer to render, however, CPU resources will be still available.");
//   } catch (vk::IncompatibleDriverError const &err) {
//     logger::error("Vulkan is incompatible with your driver. You may not use the renderer "
//                   "to render, however, CPU resources will be still available.");
//   }
// }

// static inline uint32_t computeDevicePriority(Context::PhysicalDeviceInfo const &info,
//                                              std::string const &hint, bool presentRequested) {

//   // specific cuda device
//   if (hint.starts_with("cuda:")) {
//     if (info.cudaId == std::stoi(hint.substr(5))) {
//       return 1000;
//     }
//     return 0;
//   }

//   // any cuda device
//   if (hint == std::string("cuda")) {
//     uint32_t score = 0;
//     if (info.cudaId >= 0) {
//       score += 1000;
//     }
//     if (info.present && presentRequested) {
//       score += 100;
//     }
//     return score;
//   }

//   // specific device
//   if (hint.starts_with("pci:")) {
//     if (info.pciBus == std::stoi(hint.substr(4), 0, 16)) {
//       return 1000;
//     }
//     return 0;
//   }

//   // no device hint
//   if (!info.supported) {
//     return 0;
//   }
//   uint32_t score = 0;
//   if (info.present && presentRequested) {
//     score += 100;
//   }
//   if (info.discrete) {
//     score += 10;
//   }
//   if (info.rayTracing) {
//     score += 1;
//   }
//   return score;
// }

// std::vector<Context::PhysicalDeviceInfo> Context::summarizeDeviceInfo(VkSurfaceKHR tmpSurface) {
//   std::vector<Context::PhysicalDeviceInfo> devices;

//   std::stringstream ss;
//   ss << "Devices visible to Vulkan" << std::endl;
//   ss << std::setw(3) << "Id" << std::setw(40) << "name" << std::setw(10) << "Present"
//      << std::setw(10) << "Supported" << std::setw(10) << "PciBus" << std::setw(10) << "CudaId"
//      << std::setw(15) << "RayTracing" << std::setw(10) << "Discrete" << std::endl;

//   vk::PhysicalDeviceFeatures2 features{};
//   vk::PhysicalDeviceDescriptorIndexingFeatures descriptorFeatures{};
//   vk::PhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures{};
//   features.setPNext(&descriptorFeatures);
//   descriptorFeatures.setPNext(&timelineSemaphoreFeatures);

//   int ord = 0;
//   for (auto device : mInstance->getInternal().enumeratePhysicalDevices()) {
//     std::string name;
//     bool present = false;
//     bool required_features = false;
//     int cudaId = -1;
//     int busid = -1;
//     int queueIdxNoPresent = -1;
//     int queueIdxPresent = -1;
//     int queueIdx = -1;
//     bool rayTracing = false;
//     bool discrete = false;

//     // check graphics & compute queues
//     std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
//         device.getQueueFamilyProperties();
//     for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
//       if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics &&
//           queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute) {
//         if (queueIdxNoPresent == -1) {
//           queueIdxNoPresent = i;
//         }
//         if (mPresentRequested && device.getSurfaceSupportKHR(i, tmpSurface)) {
//           queueIdxPresent = i;
//           present = true;
//           break;
//         }
//       }
//     }

//     if (queueIdxPresent != -1) {
//       queueIdx = queueIdxPresent;
//     } else {
//       queueIdx = queueIdxNoPresent;
//     }

//     // check features
//     device.getFeatures2(&features);
//     if (features.features.independentBlend && features.features.wideLines &&
//         features.features.geometryShader && descriptorFeatures.descriptorBindingPartiallyBound
//         && timelineSemaphoreFeatures.timelineSemaphore) {
//       required_features = true;
//     }

//     // check extensions
//     auto extensions = device.enumerateDeviceExtensionProperties();
//     for (auto &ext : extensions) {
//       if (std::strcmp(ext.extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) == 0) {
//         rayTracing = 1;
//       }
//     }

//     auto properties = device.getProperties();
//     name = std::string(properties.deviceName.begin(), properties.deviceName.end());
//     discrete = properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;

//     vk::PhysicalDeviceProperties2KHR p2;
//     vk::PhysicalDevicePCIBusInfoPropertiesEXT pciInfo;
//     pciInfo.pNext = p2.pNext;
//     p2.pNext = &pciInfo;
//     device.getProperties2(&p2);
//     busid = pciInfo.pciBus;

//     int computeMode{-1};
//     auto SAPIEN_DISABLE_RAY_TRACING = std::getenv("SAPIEN_DISABLE_RAY_TRACING");
//     if (SAPIEN_DISABLE_RAY_TRACING && std::strcmp(SAPIEN_DISABLE_RAY_TRACING, "1") == 0) {
//       rayTracing = 0;
//     } else {
// #ifdef SVULKAN2_CUDA_INTEROP
//       cudaId = getCudaDeviceIdFromPhysicalDevice(device);
//       cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, cudaId);
//       if ((computeMode == cudaComputeModeExclusiveProcess ||
//            computeMode == cudaComputeModeExclusive) &&
//           rayTracing) {
//         logger::warn("CUDA device {} is in EXCLUSIVE or EXCLUSIVE_PROCESS mode. You "
//                      "many not use this renderer with external CUDA programs unless "
//                      "you switch off ray tracing by environment variable "
//                      "SAPIEN_DISABLE_RAY_TRACING=1.",
//                      cudaId);
//       }
// #endif
//     }
//     bool supported = required_features && queueIdx != -1;

//     ss << std::setw(3) << ord++ << std::setw(40) << name.substr(0, 39).c_str() << std::setw(10)
//        << present << std::setw(10) << supported << std::hex << std::setw(10) << busid <<
//        std::dec
//        << std::setw(10) << (cudaId < 0 ? "No Device" : std::to_string(cudaId)) << std::setw(15)
//        << rayTracing << std::setw(10) << discrete << std::endl;

//     devices.push_back(Context::PhysicalDeviceInfo{.device = device,
//                                                   .present = present,
//                                                   .supported = required_features,
//                                                   .cudaId = cudaId,
//                                                   .pciBus = busid,
//                                                   .queueIndex = queueIdx,
//                                                   .rayTracing = rayTracing,
//                                                   .cudaComputeMode = computeMode,
//                                                   .discrete = discrete});
//   }
//   logger::info(ss.str());

// #ifdef SVULKAN2_CUDA_INTEROP
//   ss = {};
//   ss << "Devices visible to Cuda" << std::endl;
//   ss << std::setw(10) << "CudaId" << std::setw(10) << "PciBus" << std::setw(25) <<
//   "PciBusString"
//      << std::endl;

//   int count{0};
//   cudaGetDeviceCount(&count);
//   for (uint32_t i = 0; i < count; ++i) {
//     int busId = getPCIBusIdFromCudaDeviceId(i);
//     std::string pciBus(20, '\0');
//     auto result = cudaDeviceGetPCIBusId(pciBus.data(), 20, i);

//     switch (result) {
//     case cudaErrorInvalidValue:
//       pciBus = "Invalid Value";
//       break;
//     case cudaErrorInvalidDevice:
//       pciBus = "Invalid Device";
//       break;
//     case cudaErrorInitializationError:
//       pciBus = "Not Initialized";
//       break;
//     case cudaErrorInsufficientDriver:
//       pciBus = "Insufficient Driver";
//       break;
//     case cudaErrorNoDevice:
//       pciBus = "No Device";
//       break;
//     default:
//       ss << std::setw(10) << i << std::hex << std::setw(10) << busId << std::dec <<
//       std::setw(25)
//          << pciBus.c_str() << std::endl;
//       break;
//     }
//   }
//   logger::info(ss.str());
// #endif

//   return devices;
// }

// void Context::pickSuitableGpuAndQueueFamilyIndex() {
//   if (!isVulkanAvailable()) {
//     return;
//   }

//   // make a temporary GLFW window to test present
//   GLFWwindow *window{};

//   VkSurfaceKHR tmpSurface = nullptr;

//   if (mPresentRequested) {
//     glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//     glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
//     window = glfwCreateWindow(1, 1, "vulkan", nullptr, nullptr);
//     auto result = glfwCreateWindowSurface(mInstance->getInternal(), window, nullptr,
//     &tmpSurface); if (result != VK_SUCCESS) {
//       throw std::runtime_error("Window creation failed, you may not create "
//                                "GLFW window for presenting rendered results.");
//     }
//     glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
//   }

//   // summarize capabilities of all detected devices
//   auto devices = summarizeDeviceInfo(tmpSurface);

//   if (mPresentRequested) {
//     mInstance->getInternal().destroySurfaceKHR(tmpSurface, nullptr);
//     glfwDestroyWindow(window);
//   }

//   // pick device with highest priority
//   int pickedDeviceIdx = -1;
//   {
//     uint32_t maxPriority = 0;
//     uint32_t idx = 0;
//     for (auto &device : devices) {
//       uint32_t priority = computeDevicePriority(device, mDeviceHint, mPresentRequested);
//       if (priority > maxPriority) {
//         maxPriority = priority;
//         pickedDeviceIdx = idx;
//       }
//       idx++;
//     }
//   }

//   if (pickedDeviceIdx == -1) {
//     throw std::runtime_error("Cannot find a suitable rendering device");
//   }

//   if (mPresentRequested && !devices[pickedDeviceIdx].present) {
//     logger::warn("Present requested but the selected device does not support "
//                  "present. Continue without present.");
//     mPresentRequested = false;
//   }

//   logger::info("Vulkan picked device: {}", pickedDeviceIdx);

//   vk::PhysicalDevice pickedDevice = devices[pickedDeviceIdx].device;

//   mPhysicalDeviceInfo = devices[pickedDeviceIdx];
//   mPhysicalDeviceLimits = pickedDevice.getProperties().limits;
// }

// void Context::createDevice() {
//   if (!isVulkanAvailable()) {
//     return;
//   }

//   float queuePriority = 0.0f;
//   vk::DeviceQueueCreateInfo deviceQueueCreateInfo({}, getGraphicsQueueFamilyIndex(), 1,
//                                                   &queuePriority);
//   std::vector<const char *> deviceExtensions{};

//   vk::PhysicalDeviceFeatures2 features{};
//   vk::PhysicalDeviceDescriptorIndexingFeatures descriptorFeatures{};
//   vk::PhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures{};

//   features.setPNext(&descriptorFeatures);
//   descriptorFeatures.setPNext(&timelineSemaphoreFeatures);

//   features.features.setIndependentBlend(true);
//   features.features.setWideLines(true);
//   features.features.setGeometryShader(true);
//   descriptorFeatures.setDescriptorBindingPartiallyBound(true);
//   timelineSemaphoreFeatures.setTimelineSemaphore(true);
//   descriptorFeatures.setPNext(&timelineSemaphoreFeatures);

//   vk::PhysicalDeviceAccelerationStructureFeaturesKHR asFeature;
//   asFeature.setAccelerationStructure(true);
//   vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rtFeature;
//   rtFeature.setRayTracingPipeline(true);
//   vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR addrFeature;
//   addrFeature.setBufferDeviceAddress(true);
//   vk::PhysicalDeviceShaderClockFeaturesKHR clockFeature;
//   clockFeature.setShaderDeviceClock(true);
//   clockFeature.setShaderSubgroupClock(true);

//   if (isRayTracingAvailable()) {
//     deviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
//     deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
//     deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
//     deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
//     deviceExtensions.push_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
//     descriptorFeatures.setRuntimeDescriptorArray(true);
//     descriptorFeatures.setShaderStorageBufferArrayNonUniformIndexing(true);
//     descriptorFeatures.setShaderSampledImageArrayNonUniformIndexing(true);
//     timelineSemaphoreFeatures.setPNext(&asFeature);
//     asFeature.setPNext(&rtFeature);
//     rtFeature.setPNext(&addrFeature);
//     addrFeature.setPNext(&clockFeature);
//     features.features.setShaderInt64(true);
//   }

// #ifdef SVULKAN2_CUDA_INTEROP
//   deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
//   deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
//   deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
//   deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
// #endif

//   if (isPresentAvailable()) {
//     deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
//   }

//   vk::DeviceCreateInfo deviceInfo({}, deviceQueueCreateInfo, {}, deviceExtensions);
//   deviceInfo.setPNext(&features);
//   mDevice = getPhysicalDevice().createDeviceUnique(deviceInfo);
//   VULKAN_HPP_DEFAULT_DISPATCHER.init(mDevice.get());

//   mQueue = std::make_unique<Queue>();
// }

// void Context::createMemoryAllocator() {
//   if (!isVulkanAvailable()) {
//     return;
//   }

//   VmaVulkanFunctions vulkanFunctions{};
//   vulkanFunctions.vkGetInstanceProcAddr =
//       (PFN_vkGetInstanceProcAddr)mInstance->getProcAddr("vkGetInstanceProcAddr");
//   vulkanFunctions.vkGetDeviceProcAddr =
//       (PFN_vkGetDeviceProcAddr)mInstance->getProcAddr("vkGetDeviceProcAddr");
//   vulkanFunctions.vkGetPhysicalDeviceProperties =
//       (PFN_vkGetPhysicalDeviceProperties)mInstance->getProcAddr("vkGetPhysicalDeviceProperties");
//   vulkanFunctions.vkGetPhysicalDeviceMemoryProperties =
//       (PFN_vkGetPhysicalDeviceMemoryProperties)mInstance->getProcAddr(
//           "vkGetPhysicalDeviceMemoryProperties");
//   vulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR =
//       (PFN_vkGetPhysicalDeviceMemoryProperties2)mInstance->getProcAddr(
//           "vkGetPhysicalDeviceMemoryProperties2");
//   vulkanFunctions.vkAllocateMemory =
//       (PFN_vkAllocateMemory)mDevice->getProcAddr("vkAllocateMemory");
//   vulkanFunctions.vkFreeMemory = (PFN_vkFreeMemory)mDevice->getProcAddr("vkFreeMemory");
//   vulkanFunctions.vkMapMemory = (PFN_vkMapMemory)mDevice->getProcAddr("vkMapMemory");
//   vulkanFunctions.vkUnmapMemory = (PFN_vkUnmapMemory)mDevice->getProcAddr("vkUnmapMemory");
//   vulkanFunctions.vkFlushMappedMemoryRanges =
//       (PFN_vkFlushMappedMemoryRanges)mDevice->getProcAddr("vkFlushMappedMemoryRanges");
//   vulkanFunctions.vkInvalidateMappedMemoryRanges =
//       (PFN_vkInvalidateMappedMemoryRanges)mDevice->getProcAddr("vkInvalidateMappedMemoryRanges");
//   vulkanFunctions.vkBindBufferMemory =
//       (PFN_vkBindBufferMemory)mDevice->getProcAddr("vkBindBufferMemory");
//   vulkanFunctions.vkBindImageMemory =
//       (PFN_vkBindImageMemory)mDevice->getProcAddr("vkBindImageMemory");
//   vulkanFunctions.vkGetBufferMemoryRequirements =
//       (PFN_vkGetBufferMemoryRequirements)mDevice->getProcAddr("vkGetBufferMemoryRequirements");
//   vulkanFunctions.vkGetImageMemoryRequirements =
//       (PFN_vkGetImageMemoryRequirements)mDevice->getProcAddr("vkGetImageMemoryRequirements");
//   vulkanFunctions.vkCreateBuffer = (PFN_vkCreateBuffer)mDevice->getProcAddr("vkCreateBuffer");
//   vulkanFunctions.vkDestroyBuffer =
//   (PFN_vkDestroyBuffer)mDevice->getProcAddr("vkDestroyBuffer"); vulkanFunctions.vkCreateImage =
//   (PFN_vkCreateImage)mDevice->getProcAddr("vkCreateImage"); vulkanFunctions.vkDestroyImage =
//   (PFN_vkDestroyImage)mDevice->getProcAddr("vkDestroyImage"); vulkanFunctions.vkCmdCopyBuffer =
//   (PFN_vkCmdCopyBuffer)mDevice->getProcAddr("vkCmdCopyBuffer");
//   vulkanFunctions.vkGetBufferMemoryRequirements2KHR =
//       (PFN_vkGetBufferMemoryRequirements2KHR)mDevice->getProcAddr(
//           "vkGetBufferMemoryRequirements2");
//   vulkanFunctions.vkGetImageMemoryRequirements2KHR =
//       (PFN_vkGetImageMemoryRequirements2KHR)mDevice->getProcAddr("vkGetImageMemoryRequirements2");
//   vulkanFunctions.vkBindBufferMemory2KHR =
//       (PFN_vkBindBufferMemory2KHR)mDevice->getProcAddr("vkBindBufferMemory2");
//   vulkanFunctions.vkBindImageMemory2KHR =
//       (PFN_vkBindImageMemory2KHR)mDevice->getProcAddr("vkBindImageMemory2");

//   VmaAllocatorCreateInfo allocatorInfo = {};
//   allocatorInfo.vulkanApiVersion = mInstance->getApiVersion();
//   allocatorInfo.physicalDevice = getPhysicalDevice();
//   allocatorInfo.device = mDevice.get();
//   allocatorInfo.instance = mInstance->getInternal();
//   allocatorInfo.pVulkanFunctions = &vulkanFunctions;

//   if (isRayTracingAvailable()) {
//     allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
//   }

//   mAllocator = std::make_unique<Allocator>(allocatorInfo);
// }

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
