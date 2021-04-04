#include "svulkan2/core/descriptor_pool.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace core {

DynamicDescriptorPool::DynamicDescriptorPool(
    std::shared_ptr<Context> context,
    std::vector<vk::DescriptorPoolSize> const &sizes)
    : mContext(context), mSizes(sizes) {
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
  mPools.push_back(mContext->getDevice().createDescriptorPoolUnique(info));
}

vk::UniqueDescriptorSet
DynamicDescriptorPool::allocateSet(vk::DescriptorSetLayout layout) {
  for (auto &pool : mPools) {
    try {
      return std::move(
          mContext->getDevice()
              .allocateDescriptorSetsUnique(
                  vk::DescriptorSetAllocateInfo(pool.get(), 1, &layout))
              .front());
    } catch (vk::OutOfPoolMemoryError const &) {
    }
  }
  expand();
  return std::move(
      mContext->getDevice()
          .allocateDescriptorSetsUnique(
              vk::DescriptorSetAllocateInfo(mPools.back().get(), 1, &layout))
          .front());
}

} // namespace core
} // namespace svulkan2
