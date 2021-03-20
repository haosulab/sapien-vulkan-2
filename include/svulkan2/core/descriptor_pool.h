#pragma once
#include "svulkan2/common/vk.h"
#include <memory>

namespace svulkan2 {
namespace core {

class DynamicDescriptorPool {
  class Context *mContext;
  std::vector<vk::DescriptorPoolSize> mSizes;
  std::vector<vk::UniqueDescriptorPool> mPools;

public:
  DynamicDescriptorPool(class Context &context,
                        std::vector<vk::DescriptorPoolSize> const &sizes);
  DynamicDescriptorPool(DynamicDescriptorPool &other) = delete;
  DynamicDescriptorPool(DynamicDescriptorPool &&other) = default;
  DynamicDescriptorPool &operator=(DynamicDescriptorPool &other) = delete;
  DynamicDescriptorPool &operator=(DynamicDescriptorPool &&other) = default;

  vk::UniqueDescriptorSet allocateSet(vk::DescriptorSetLayout layout);

private:
  void expand();
};

} // namespace core
} // namespace svulkan2
