#include "svulkan2/resource/render_target.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace resource {

SVRenderTarget::SVRenderTarget(std::string const &name, uint32_t width,
                               uint32_t height, vk::Format format)
    : mName(name), mFormat(format), mWidth(width), mHeight(height) {}

void SVRenderTarget::createDeviceResources(core::Context &context) {
  bool isDepth = false;
  vk::ImageUsageFlags usage;
  if (mFormat == vk::Format::eR8G8B8A8Unorm ||
      mFormat == vk::Format::eR32G32B32A32Sfloat ||
      mFormat == vk::Format::eR32G32B32A32Uint) {
    usage = vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eColorAttachment |
            vk::ImageUsageFlagBits::eTransferSrc;
  } else if (mFormat == vk::Format::eD32Sfloat) {
    usage = vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eDepthStencilAttachment |
            vk::ImageUsageFlagBits::eTransferSrc;
    isDepth = true;
  } else {
    throw std::runtime_error(
        "failed to create image resources: unsupported image format. Currently "
        "supported formats are R32G32B32A32Sfloat, R8G8B8A8Unorm, "
        "R32G32B32A32Uint, D32Sfloat, D24UnormS8Uint");
  }

  mImage = std::make_unique<core::Image>(
      context, vk::Extent3D{mWidth, mHeight, 1}, mFormat, usage,
      VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, vk::SampleCountFlagBits::e1,
      1);
  vk::ComponentMapping componentMapping(
      vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
      vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA);

  vk::ImageViewCreateInfo info(
      {}, mImage->getVulkanImage(), vk::ImageViewType::e2D, mFormat,
      componentMapping,
      vk::ImageSubresourceRange(isDepth ? vk::ImageAspectFlagBits::eDepth
                                        : vk::ImageAspectFlagBits::eColor,
                                0, 1, 0, 1));
  mImageView = context.getDevice().createImageViewUnique(info);
  mSampler = context.getDevice().createSamplerUnique(vk::SamplerCreateInfo(
      {}, vk::Filter::eNearest, vk::Filter::eNearest,
      vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToBorder,
      vk::SamplerAddressMode::eClampToBorder,
      vk::SamplerAddressMode::eClampToBorder, 0.f, false, 0.f, false,
      vk::CompareOp::eNever, 0.f, 0.f, vk::BorderColor::eFloatOpaqueBlack));
}

} // namespace resource
} // namespace svulkan2
