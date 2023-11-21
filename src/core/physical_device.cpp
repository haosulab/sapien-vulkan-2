#include "svulkan2/core/physical_device.h"
#include "../common/cuda_helper.h"
#include "../common/logger.h"
#include "svulkan2/core/device.h"
#include "svulkan2/core/instance.h"
#include <GLFW/glfw3.h>
#include <iomanip>
#include <sstream>

namespace svulkan2 {
namespace core {

static inline uint32_t computeDevicePriority(PhysicalDevice::DeviceInfo const &info,
                                             std::string const &hint, bool presentRequested) {

  // specific cuda device
  if (hint.starts_with("cuda:")) {
    if (info.cudaId == std::stoi(hint.substr(5))) {
      return 1000;
    }
    return 0;
  }

  // any cuda device
  if (hint == std::string("cuda")) {
    uint32_t score = 0;
    if (info.cudaId >= 0) {
      score += 1000;
    }
    if (info.present && presentRequested) {
      score += 100;
    }
    if (info.rayTracing) {
      score += 1;
    }
    return score;
  }

  // specific device
  if (hint.starts_with("pci:")) {
    if (info.pciBus == std::stoi(hint.substr(4), 0, 16)) {
      return 1000;
    }
    return 0;
  }

  // no device hint
  // still prefer cuda device
  if (!info.supported) {
    return 0;
  }
  uint32_t score = 0;
  if (info.cudaId >= 0) {
    score += 1000;
  }
  if (info.present && presentRequested) {
    score += 100;
  }
  if (info.discrete) {
    score += 10;
  }
  if (info.rayTracing) {
    score += 1;
  }
  return score;
}

PhysicalDevice::PhysicalDevice(std::shared_ptr<Instance> instance, std::string const &hint)
    : mInstance(instance) {
  if (!instance) {
    throw std::runtime_error("invalid instance");
  }

  auto devices = summarizeDeviceInfo();

  // pick device with highest priority
  int pickedDeviceIdx = -1;
  {
    uint32_t maxPriority = 0;
    uint32_t idx = 0;
    for (auto &device : devices) {
      uint32_t priority = computeDevicePriority(device, hint, mInstance->isGLFWEnabled());
      if (priority > maxPriority) {
        maxPriority = priority;
        pickedDeviceIdx = idx;
      }
      idx++;
    }
  }

  if (pickedDeviceIdx == -1) {
    throw std::runtime_error("Failed to find a supported rendering device.");
  }

  logger::info("Picked Vulkan device: {}", pickedDeviceIdx);
  vk::PhysicalDevice pickedDevice = devices[pickedDeviceIdx].device;

  mPickedDeviceInfo = devices[pickedDeviceIdx];
  mPickedDeviceLimits = pickedDevice.getProperties().limits;
}

std::shared_ptr<Device> PhysicalDevice::createDevice() {
  return std::make_shared<Device>(shared_from_this());
}

std::vector<PhysicalDevice::DeviceInfo> PhysicalDevice::summarizeDeviceInfo() const {
  std::vector<PhysicalDevice::DeviceInfo> devices;

  // prepare a surface for window testing
  bool presentEnabled = mInstance->isGLFWEnabled();
  GLFWwindow *window{};
  VkSurfaceKHR tmpSurface = nullptr;
  if (presentEnabled) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(1, 1, "SAPIEN", nullptr, nullptr);
    auto result = glfwCreateWindowSurface(mInstance->getInternal(), window, nullptr, &tmpSurface);
    if (result != VK_SUCCESS) {
      throw std::runtime_error(
          "Window creation test failed, you may not create GLFW window for display.");
      presentEnabled = false;
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
  }

  std::stringstream ss;
  ss << "Devices visible to Vulkan" << std::endl;
  ss << std::setw(3) << "Id" << std::setw(40) << "name" << std::setw(10) << "Present"
     << std::setw(10) << "Supported" << std::setw(10) << "PciBus" << std::setw(10) << "CudaId"
     << std::setw(15) << "RayTracing" << std::setw(10) << "Discrete" << std::endl;

  vk::PhysicalDeviceFeatures2 features{};
  vk::PhysicalDeviceDescriptorIndexingFeatures descriptorFeatures{};
  vk::PhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures{};
  features.setPNext(&descriptorFeatures);
  descriptorFeatures.setPNext(&timelineSemaphoreFeatures);

  int ord = 0;
  for (auto device : mInstance->getInternal().enumeratePhysicalDevices()) {
    std::string name;
    bool present = false;
    bool required_features = false;
    int cudaId = -1;
    int busid = -1;
    int queueIdxNoPresent = -1;
    int queueIdxPresent = -1;
    int queueIdx = -1;
    bool rayTracing = false;
    bool discrete = false;

    // check graphics & compute queues
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
        device.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
      if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics &&
          queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute) {
        if (queueIdxNoPresent == -1) {
          queueIdxNoPresent = i;
        }
        if (presentEnabled && device.getSurfaceSupportKHR(i, tmpSurface)) {
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

    // check features
    device.getFeatures2(&features);
    if (features.features.independentBlend && features.features.wideLines &&
        features.features.geometryShader && descriptorFeatures.descriptorBindingPartiallyBound &&
        timelineSemaphoreFeatures.timelineSemaphore) {
      required_features = true;
    }

    std::vector<const char *> all_required_extensions = {
        VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME};
    size_t required_extension_count = 0;

    // check extensions
    auto extensions = device.enumerateDeviceExtensionProperties();
    for (auto &ext : extensions) {
      if (std::strcmp(ext.extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) == 0) {
        rayTracing = 1;
      }
      for (auto name : all_required_extensions)
        if (std::strcmp(ext.extensionName, name) == 0) {
          required_extension_count += 1;
        }
    }

    bool required_extensions = required_extension_count == all_required_extensions.size();

    auto properties = device.getProperties();
    name = std::string(properties.deviceName.begin(), properties.deviceName.end());
    discrete = properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;

    vk::PhysicalDeviceProperties2KHR p2;
    vk::PhysicalDevicePCIBusInfoPropertiesEXT pciInfo;
    pciInfo.pNext = p2.pNext;
    p2.pNext = &pciInfo;
    device.getProperties2(&p2);
    busid = pciInfo.pciBus;

    int computeMode{-1};
    auto SAPIEN_DISABLE_RAY_TRACING = std::getenv("SAPIEN_DISABLE_RAY_TRACING");
    if (SAPIEN_DISABLE_RAY_TRACING && std::strcmp(SAPIEN_DISABLE_RAY_TRACING, "1") == 0) {
      rayTracing = 0;
    } else {
#ifdef SVULKAN2_CUDA_INTEROP
      cudaId = getCudaDeviceIdFromPhysicalDevice(device);
      cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, cudaId);
      if ((computeMode == cudaComputeModeExclusiveProcess ||
           computeMode == cudaComputeModeExclusive) &&
          rayTracing) {
        logger::warn("CUDA device {} is in EXCLUSIVE or EXCLUSIVE_PROCESS mode. You "
                     "many not use this renderer with external CUDA programs unless "
                     "you switch off ray tracing by environment variable "
                     "SAPIEN_DISABLE_RAY_TRACING=1.",
                     cudaId);
      }
#endif
    }
    bool supported = required_features && required_extensions && queueIdx != -1;

    ss << std::setw(3) << ord++ << std::setw(40) << name.substr(0, 39).c_str() << std::setw(10)
       << present << std::setw(10) << supported << std::hex << std::setw(10) << busid << std::dec
       << std::setw(10) << (cudaId < 0 ? "No Device" : std::to_string(cudaId)) << std::setw(15)
       << rayTracing << std::setw(10) << discrete << std::endl;

    devices.push_back(PhysicalDevice::DeviceInfo{.device = device,
                                                 .present = present,
                                                 .supported = supported,
                                                 .cudaId = cudaId,
                                                 .pciBus = busid,
                                                 .queueIndex = queueIdx,
                                                 .rayTracing = rayTracing,
                                                 .cudaComputeMode = computeMode,
                                                 .discrete = discrete});
  }
  logger::info(ss.str());

#ifdef SVULKAN2_CUDA_INTEROP
  ss = {};
  ss << "Devices visible to Cuda" << std::endl;
  ss << std::setw(10) << "CudaId" << std::setw(10) << "PciBus" << std::setw(25) << "PciBusString"
     << std::endl;

  int count{0};
  cudaGetDeviceCount(&count);
  for (int i = 0; i < count; ++i) {
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
      ss << std::setw(10) << i << std::hex << std::setw(10) << busId << std::dec << std::setw(25)
         << pciBus.c_str() << std::endl;
      break;
    }
  }
  logger::info(ss.str());
#endif

  // clean up
  if (tmpSurface) {
    mInstance->getInternal().destroySurfaceKHR(tmpSurface);
  }
  if (window) {
    glfwDestroyWindow(window);
  }

  return devices;
}

vk::PhysicalDeviceRayTracingPipelinePropertiesKHR PhysicalDevice::getRayTracingProperties() const {
  if (!mPickedDeviceInfo.rayTracing) {
    throw std::runtime_error("the physical device does not support ray tracing");
  }

  auto properties =
      mPickedDeviceInfo.device.getProperties2<vk::PhysicalDeviceProperties2,
                                              vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
  return properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
}

vk::PhysicalDeviceAccelerationStructurePropertiesKHR PhysicalDevice::getASProperties() const {
  if (!mPickedDeviceInfo.rayTracing) {
    throw std::runtime_error("the physical device does not support ray tracing");
  }
  auto properties = mPickedDeviceInfo.device.getProperties2<
      vk::PhysicalDeviceProperties2, vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
  return properties.get<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
}

PhysicalDevice::~PhysicalDevice() {}

} // namespace core
} // namespace svulkan2
