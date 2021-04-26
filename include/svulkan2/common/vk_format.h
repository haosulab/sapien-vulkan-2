#pragma once

#include <vulkan/vulkan.hpp>

namespace svulkan2 {

enum VulkanBaseType {
  eI8,
  eI16,
  eI32,
  eI64,
  eU8,
  eU16,
  eU32,
  eU64,
  eF16,
  eF32,
  eF64,
  eMixed
};

struct VulkanFormatInfo {
  uint32_t size;
  uint32_t channels;
  uint32_t elemSize;
  VulkanBaseType elemType;
  vk::ImageAspectFlags aspect;
};

uint32_t getFormatSize(vk::Format format);
uint32_t getFormatElementSize(vk::Format format);
uint32_t getFormatChannels(vk::Format format);
VulkanBaseType getFormatElementType(vk::Format format);
vk::ImageAspectFlags getImageAspectFlags(vk::Format format);

template <typename T> bool isFormatCompatible(vk::Format format);

} // namespace svulkan2
