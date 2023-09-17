#pragma once

#include <vulkan/vulkan.hpp>

namespace svulkan2 {

enum class ComponentFormat { eSnorm, eUnorm, eSint, eUint, eSfloat, eOther };
enum class ComponentBits { e8, e16, e32, e64, eOther };

struct VulkanFormatInfo {
  uint32_t size;
  uint32_t channels;
  uint32_t elemSize;

  ComponentFormat format;
  ComponentBits bits;
  bool srgb;

  vk::ImageAspectFlags aspect;
};

uint32_t getFormatSize(vk::Format format);
uint32_t getFormatChannels(vk::Format format);
uint32_t getFormatElementSize(vk::Format format);
ComponentFormat getFormatComponentFormat(vk::Format format);
ComponentBits getFormatComponentBits(vk::Format format);

// VulkanBaseType getFormatElementType(vk::Format format);

vk::ImageAspectFlags getImageAspectFlags(vk::Format format);

template <typename T> bool isFormatCompatible(vk::Format format);

} // namespace svulkan2
