#include "svulkan2/common/vk_format.h"
#include <map>

namespace svulkan2 {

const std::map<vk::Format, VulkanFormatInfo> gFormatTable = {
    // Uint
    {vk::Format::eR8Uint,
     {1, 1, 1, VulkanBaseType::eU8, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR8G8Uint,
     {2, 2, 1, VulkanBaseType::eU8, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR8G8B8Uint,
     {3, 3, 1, VulkanBaseType::eU8, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR8G8B8A8Uint,
     {4, 4, 1, VulkanBaseType::eU8, vk::ImageAspectFlagBits::eColor}},

    {vk::Format::eR16Uint,
     {2, 1, 2, VulkanBaseType::eU16, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR16G16Uint,
     {4, 2, 2, VulkanBaseType::eU16, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR16G16B16Uint,
     {6, 3, 2, VulkanBaseType::eU16, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR16G16B16A16Uint,
     {8, 4, 2, VulkanBaseType::eU16, vk::ImageAspectFlagBits::eColor}},

    {vk::Format::eR32Uint,
     {4, 1, 4, VulkanBaseType::eU32, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR32G32Uint,
     {8, 2, 4, VulkanBaseType::eU32, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR32G32B32Uint,
     {12, 3, 4, VulkanBaseType::eU32, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR32G32B32A32Uint,
     {16, 4, 4, VulkanBaseType::eU32, vk::ImageAspectFlagBits::eColor}},

    {vk::Format::eR64Uint,
     {8, 1, 8, VulkanBaseType::eU64, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR64G64Uint,
     {16, 2, 8, VulkanBaseType::eU64, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR64G64B64Uint,
     {24, 3, 8, VulkanBaseType::eU64, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR64G64B64A64Uint,
     {32, 4, 8, VulkanBaseType::eU64, vk::ImageAspectFlagBits::eColor}},

    // int
    {vk::Format::eR8Sint,
     {1, 1, 1, VulkanBaseType::eI8, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR8G8Sint,
     {2, 2, 1, VulkanBaseType::eI8, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR8G8B8Sint,
     {3, 3, 1, VulkanBaseType::eI8, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR8G8B8A8Sint,
     {4, 4, 1, VulkanBaseType::eI8, vk::ImageAspectFlagBits::eColor}},

    {vk::Format::eR16Sint,
     {2, 1, 2, VulkanBaseType::eI16, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR16G16Sint,
     {4, 2, 2, VulkanBaseType::eI16, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR16G16B16Sint,
     {6, 3, 2, VulkanBaseType::eI16, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR16G16B16A16Sint,
     {8, 4, 2, VulkanBaseType::eI16, vk::ImageAspectFlagBits::eColor}},

    {vk::Format::eR32Sint,
     {4, 1, 4, VulkanBaseType::eI32, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR32G32Sint,
     {8, 2, 4, VulkanBaseType::eI32, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR32G32B32Sint,
     {12, 3, 4, VulkanBaseType::eI32, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR32G32B32A32Sint,
     {16, 4, 4, VulkanBaseType::eI32, vk::ImageAspectFlagBits::eColor}},

    {vk::Format::eR64Sint,
     {8, 1, 8, VulkanBaseType::eI64, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR64G64Sint,
     {16, 2, 8, VulkanBaseType::eI64, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR64G64B64Sint,
     {24, 3, 8, VulkanBaseType::eI64, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR64G64B64A64Sint,
     {32, 4, 8, VulkanBaseType::eI64, vk::ImageAspectFlagBits::eColor}},

    // Unorm
    {vk::Format::eR8Unorm,
     {1, 1, 1, VulkanBaseType::eU8, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR8G8Unorm,
     {2, 2, 1, VulkanBaseType::eU8, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR8G8B8Unorm,
     {3, 3, 1, VulkanBaseType::eU8, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR8G8B8A8Unorm,
     {4, 4, 1, VulkanBaseType::eU8, vk::ImageAspectFlagBits::eColor}},

    {vk::Format::eR16Unorm,
     {2, 1, 2, VulkanBaseType::eU16, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR16G16Unorm,
     {4, 2, 2, VulkanBaseType::eU16, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR16G16B16Unorm,
     {6, 3, 2, VulkanBaseType::eU16, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR16G16B16A16Unorm,
     {8, 4, 2, VulkanBaseType::eU16, vk::ImageAspectFlagBits::eColor}},

    // Sfloat
    {vk::Format::eR16Sfloat,
     {2, 1, 2, VulkanBaseType::eF16, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR16G16Sfloat,
     {4, 2, 2, VulkanBaseType::eF16, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR16G16B16Sfloat,
     {6, 3, 2, VulkanBaseType::eF16, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR16G16B16A16Sfloat,
     {8, 4, 2, VulkanBaseType::eF16, vk::ImageAspectFlagBits::eColor}},

    {vk::Format::eR32Sfloat,
     {4, 1, 4, VulkanBaseType::eF32, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR32G32Sfloat,
     {8, 2, 4, VulkanBaseType::eF32, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR32G32B32Sfloat,
     {12, 3, 4, VulkanBaseType::eF32, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR32G32B32A32Sfloat,
     {16, 4, 4, VulkanBaseType::eF32, vk::ImageAspectFlagBits::eColor}},

    {vk::Format::eR64Sfloat,
     {8, 1, 8, VulkanBaseType::eF64, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR64G64Sfloat,
     {16, 2, 8, VulkanBaseType::eF64, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR64G64B64Sfloat,
     {24, 3, 8, VulkanBaseType::eF64, vk::ImageAspectFlagBits::eColor}},
    {vk::Format::eR64G64B64A64Sfloat,
     {32, 4, 8, VulkanBaseType::eF64, vk::ImageAspectFlagBits::eColor}},

    // depth formats
    {vk::Format::eD16Unorm,
     {2, 1, 2, VulkanBaseType::eU16, vk::ImageAspectFlagBits::eDepth}},
    {vk::Format::eD32Sfloat,
     {4, 1, 4, VulkanBaseType::eF32, vk::ImageAspectFlagBits::eDepth}},
    {vk::Format::eD16UnormS8Uint,
     {3, 2, 0, VulkanBaseType::eMixed,
      vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil}},
    {vk::Format::eD24UnormS8Uint,
     {4, 2, 0, VulkanBaseType::eMixed,
      vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil}},
    {vk::Format::eD32SfloatS8Uint,
     {8, 2, 0, VulkanBaseType::eMixed,
      vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil}},
};

uint32_t getFormatSize(vk::Format format) {
  return gFormatTable.at(format).size;
}

uint32_t getFormatElementSize(vk::Format format) {
  return gFormatTable.at(format).elemSize;
}

uint32_t getFormatChannels(vk::Format format) {
  return gFormatTable.at(format).channels;
}

VulkanBaseType getFormatElementType(vk::Format format) {
  return gFormatTable.at(format).elemType;
}

vk::ImageAspectFlags getImageAspectFlags(vk::Format format) {
  return gFormatTable.at(format).aspect;
}

template <> bool isFormatCompatible<int8_t>(vk::Format format) {
  return getFormatElementType(format) == VulkanBaseType::eI8;
}

template <> bool isFormatCompatible<int16_t>(vk::Format format) {
  return getFormatElementType(format) == VulkanBaseType::eI16;
}

template <> bool isFormatCompatible<int32_t>(vk::Format format) {
  return getFormatElementType(format) == VulkanBaseType::eI32;
}

template <> bool isFormatCompatible<int64_t>(vk::Format format) {
  return getFormatElementType(format) == VulkanBaseType::eI64;
}

template <> bool isFormatCompatible<uint8_t>(vk::Format format) {
  return getFormatElementType(format) == VulkanBaseType::eU8;
}

template <> bool isFormatCompatible<uint16_t>(vk::Format format) {
  return getFormatElementType(format) == VulkanBaseType::eU16;
}

template <> bool isFormatCompatible<uint32_t>(vk::Format format) {
  return getFormatElementType(format) == VulkanBaseType::eU32;
}

template <> bool isFormatCompatible<uint64_t>(vk::Format format) {
  return getFormatElementType(format) == VulkanBaseType::eU64;
}

template <> bool isFormatCompatible<float>(vk::Format format) {
  return getFormatElementType(format) == VulkanBaseType::eF32;
}

template <> bool isFormatCompatible<double>(vk::Format format) {
  return getFormatElementType(format) == VulkanBaseType::eF64;
}

} // namespace svulkan2
