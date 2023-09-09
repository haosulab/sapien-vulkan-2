#include "svulkan2/core/descriptor_pool.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace core {

DynamicDescriptorPool::DynamicDescriptorPool(
    std::vector<vk::DescriptorPoolSize> const &sizes)
    : mSizes(sizes) {
  // mContext = Context::Get();
  expand();
}

void DynamicDescriptorPool::expand() {
  std::vector<vk::DescriptorPoolSize> sizes = mSizes;
  uint32_t count = 0;
  for (auto &size : mSizes) {
    size.setDescriptorCount(size.descriptorCount * (mPools.size() + 1));
    count += size.descriptorCount;
  }
  auto info = vk::DescriptorPoolCreateInfo(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, count, sizes.size(),
      sizes.data());
  mPools.push_back(
      Context::Get()->getDevice().createDescriptorPoolUnique(info));
}

vk::UniqueDescriptorSet
DynamicDescriptorPool::allocateSet(vk::DescriptorSetLayout const &layout) {
  for (auto &pool : mPools) {
    try {
      return std::move(
          Context::Get()
              ->getDevice()
              .allocateDescriptorSetsUnique(
                  vk::DescriptorSetAllocateInfo(pool.get(), 1, &layout))
              .front());
    } catch (vk::OutOfPoolMemoryError const &) {
    }
  }
  expand();
  return std::move(
      Context::Get()
          ->getDevice()
          .allocateDescriptorSetsUnique(
              vk::DescriptorSetAllocateInfo(mPools.back().get(), 1, &layout))
          .front());
}

} // namespace core
} // namespace svulkan2
