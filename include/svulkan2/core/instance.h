#pragma once
#include "svulkan2/common/vk.h"
#include <memory>

namespace svulkan2 {
namespace core {

class PhysicalDevice;

class Instance : public std::enable_shared_from_this<Instance> {
public:
  static std::shared_ptr<Instance> Create(bool enableGLFW, uint32_t appVersion,
                                          uint32_t engineVersion,
                                          uint32_t apiVersion = VK_API_VERSION_1_2);

  Instance(bool enableGLFW, uint32_t appVersion, uint32_t engineVersion, uint32_t apiVersion);
  ~Instance();

  inline bool isGLFWEnabled() const { return mGLFWSupported; }
  vk::Instance getInternal() const { return mInstance.get(); }
  inline uint32_t getApiVersion() const { return mApiVersion; }

  auto getProcAddr(char const *name) const { return mInstance->getProcAddr(name); }

  std::shared_ptr<PhysicalDevice> createPhysicalDevice(std::string const &hint);

  Instance(Instance const &other) = delete;
  Instance(Instance const &&other) = delete;
  Instance &operator=(Instance const &other) = delete;
  Instance &operator=(Instance const &&other) = delete;

private:
  uint32_t mApiVersion;
  std::unique_ptr<vk::DynamicLoader> mDynamicLoader;
  vk::UniqueInstance mInstance{};
  bool mGLFWSupported{};
};

} // namespace core
} // namespace svulkan2
