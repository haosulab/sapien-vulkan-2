#include "svulkan2/resource/render_target.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace resource {

SVRenderTarget::SVRenderTarget(std::string const &name, uint32_t width,
                               uint32_t height, vk::Format format)
    : mName(name), mFormat(format), mWidth(width), mHeight(height) {}

void SVRenderTarget::createDeviceResources(core::Context &context) {
  vk::ImageUsageFlags usage;
  if (mFormat == vk::Format::eR8G8B8A8Unorm ||
      mFormat == vk::Format::eR32G32B32A32Sfloat) {
    usage = vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eColorAttachment |
            vk::ImageUsageFlagBits::eTransferSrc;
  } else if (mFormat == vk::Format::eD32Sfloat ||
             mFormat == vk::Format::eD24UnormS8Uint) {
    usage = vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eDepthStencilAttachment |
            vk::ImageUsageFlagBits::eTransferSrc;
  } else {
    throw std::runtime_error(
        "failed to create image resources: unsupported image format. Currently "
        "supported formats are R32G32B32A32Sfloat, R8G8B8A8Unorm, D32Sfloat, "
        "D24UnormS8Uint");
  }

  mImage = std::make_unique<core::Image>(
      context, vk::Extent3D{mWidth, mHeight, 1}, mFormat, usage,
      VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, vk::SampleCountFlagBits::e1,
      1);
}

} // namespace resource
} // namespace svulkan2
