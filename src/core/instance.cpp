#include "svulkan2/core/instance.h"
#include "../common/logger.h"
#include "svulkan2/core/physical_device.h"
#include <GLFW/glfw3.h>
#include <filesystem>

namespace svulkan2 {
namespace core {

static void glfwErrorCallback(int error_code, const char *description) {
  logger::error("GLFW error: {}. You may suppress this message by unsetting environment variable "
                "DISPLAY so SAPIEN will not attempt to test on-screen rendering",
                description);
}

std::shared_ptr<Instance> Instance::Create(uint32_t appVersion, uint32_t engineVersion,
                                           uint32_t apiVersion) {
  try {
    return std::make_shared<Instance>(VK_MAKE_VERSION(0, 0, 1), VK_MAKE_VERSION(0, 0, 1),
                                      VK_API_VERSION_1_2);
  } catch (std::runtime_error &e) {
    logger::error("Failed to create renderer with error: {}", e.what());
  }
  return nullptr;
}

Instance::Instance(uint32_t appVersion, uint32_t engineVersion, uint32_t apiVersion)
    : mApiVersion(apiVersion) {

  bool glfw = false;
  if (glfwInit() && glfwVulkanSupported()) {
    logger::info("GLFW initialized.");
    glfw = true;
    glfwSetErrorCallback(glfwErrorCallback);
  } else {
    logger::info("Failed to initialize GLFW. Display is disabled.");
  }

  // try to load system vulkan library
  try {
    mDynamicLoader = std::make_unique<vk::DynamicLoader>();
  } catch (std::runtime_error &) {
    mDynamicLoader.reset();
  }

  // try to load SAPIEN-provided Vulkan
  if (!mDynamicLoader) {
    auto path = std::getenv("SAPIEN_VULKAN_LIBRARY_PATH");
    if (path && std::filesystem::is_regular_file(std::filesystem::path(path))) {
      try {
        mDynamicLoader = std::make_unique<vk::DynamicLoader>(path);
      } catch (std::runtime_error &) {
        mDynamicLoader.reset();
      }
    }
  }

  // give up
  if (!mDynamicLoader) {
    throw std::runtime_error("Failed to load vulkan library!");
  }

  vk::ApplicationInfo appInfo("SAPIEN", appVersion, "SAPIEN", engineVersion, apiVersion);

  auto vkGetInstanceProcAddr =
      mDynamicLoader->getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

  // ========== set up layers ========== //
  std::vector<const char *> layers;
  auto vkEnumerateInstanceLayerProperties =
      mDynamicLoader->getProcAddress<PFN_vkEnumerateInstanceLayerProperties>(
          "vkEnumerateInstanceLayerProperties");

  // query all layers
  uint32_t availableLayerCount{};
  vkEnumerateInstanceLayerProperties(&availableLayerCount, nullptr);
  std::vector<VkLayerProperties> availableLayers(availableLayerCount);
  vkEnumerateInstanceLayerProperties(&availableLayerCount, availableLayers.data());

#ifdef VK_VALIDATION
  static const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
  for (const char *layerName : validationLayers) {
    bool layerFound = false;
    for (const auto &layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layers.push_back(layerName);
        layerFound = true;
        break;
      }
    }
    if (!layerFound) {
      logger::error("Validation layer " + std::string(layerName) +
                    " is not available and skipped.");
    }
  }
#endif

  // ========== set up extensions ========== //
  std::vector<const char *> extensions;
  auto vkEnumerateInstanceExtensionProperties =
      mDynamicLoader->getProcAddress<PFN_vkEnumerateInstanceExtensionProperties>(
          "vkEnumerateInstanceExtensionProperties");

  uint32_t availableExtensionCount{};
  vkEnumerateInstanceExtensionProperties(nullptr, &availableExtensionCount, nullptr);
  std::vector<VkExtensionProperties> availableExtensions(availableExtensionCount);
  vkEnumerateInstanceExtensionProperties(nullptr, &availableExtensionCount,
                                         availableExtensions.data());

  // external memory
  static const std::vector<char const *> externalMemoryExtensions = {
      VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME};
  for (const char *extName : externalMemoryExtensions) {
    if (std::any_of(availableExtensions.begin(), availableExtensions.end(),
                    [=](auto const &prop) { return strcmp(prop.extensionName, extName) == 0; })) {
      extensions.push_back(extName);
    } else {
      logger::error("Extension " + std::string(extName) +
                    " is not available. Interop with external library is disabled.");
    }
  }

  if (glfw) {
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (glfwExtensions) {
      mGLFWSupported = true;
      for (uint32_t i = 0; i < glfwExtensionCount; ++i) {
        extensions.push_back(glfwExtensions[i]);
      }
    }
  }

  try {
    mInstance = vk::createInstanceUnique(vk::InstanceCreateInfo({}, &appInfo, layers, extensions));
    VULKAN_HPP_DEFAULT_DISPATCHER.init(mInstance.get());
  } catch (vk::OutOfHostMemoryError const &err) {
    logger::error("Failed to initialize renderer: out of host memory.");
    throw err;
  } catch (vk::OutOfDeviceMemoryError const &err) {
    logger::error("Failed to initialize renderer: out of device memory.");
    throw err;
  } catch (vk::InitializationFailedError const &err) {
    logger::error("Failed to initialize Vulkan. You may not use the renderer for rendering.");
    throw err;
  } catch (vk::LayerNotPresentError const &err) {
    logger::error(
        "Failed to load required Vulkan layer. You may not use the renderer for rendering.");
    throw err;
  } catch (vk::ExtensionNotPresentError const &err) {
    logger::error(
        "Failed to load required Vulkan extension. You may not use the renderer for rendering.");
    throw err;
  } catch (vk::IncompatibleDriverError const &err) {
    logger::error(
        "Your GPU driver does not support Vulkan. You may not use the renderer for rendering.");
    throw err;
  }
}

std::shared_ptr<PhysicalDevice> Instance::createPhysicalDevice(std::string const &hint) {
  try {
    return std::make_shared<PhysicalDevice>(shared_from_this(), hint);
  } catch (std::runtime_error &e) {
    logger::error("Failed to create renderer with error: {}", e.what());
  }
  return nullptr;
}

Instance::~Instance() { glfwTerminate(); }

} // namespace core
} // namespace svulkan2
