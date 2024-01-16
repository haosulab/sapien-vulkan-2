#include "svulkan2/ui/image.h"
#include "svulkan2/core/context.h"
#include <imgui.h>
#include <imgui_impl_vulkan.h>

namespace svulkan2 {
namespace ui {

std::shared_ptr<DisplayImage> DisplayImage::Image(core::Image &image, vk::ImageLayout layout) {
  if (mImage == &image) {
    return std::static_pointer_cast<DisplayImage>(shared_from_this());
  }

  if (mDS) {
    core::Context::Get()->getDevice().waitIdle();
    mCommandBuffer.reset();
    ImGui_ImplVulkan_RemoveTexture(mDS);
    mDS = {};
  }

  mImage = &image;
  mLayout = layout;

  if (!mCommandPool) {
    mCommandPool = core::Context::Get()->createCommandPool();
  }

  mCommandBuffer = mCommandPool->allocateCommandBuffer();
  mCommandBuffer->begin({vk::CommandBufferUsageFlags()});
  mImage->transitionLayout(mCommandBuffer.get(), mLayout, vk::ImageLayout::eShaderReadOnlyOptimal,
                           vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eShaderRead,
                           vk::PipelineStageFlagBits::eAllCommands,
                           vk::PipelineStageFlagBits::eFragmentShader);
  mCommandBuffer->end();

  auto context = core::Context::Get();
  mSampler = context->createSampler(vk::SamplerCreateInfo(
      {}, vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest,
      vk::SamplerAddressMode::eClampToBorder, vk::SamplerAddressMode::eClampToBorder,
      vk::SamplerAddressMode::eClampToBorder, 0.f, false, 0.f, false, vk::CompareOp::eNever, 0.f,
      0.f, vk::BorderColor::eFloatOpaqueWhite));

  auto format = image.getFormat();
  vk::ComponentMapping mapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                               vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA);
  if (getFormatChannels(format) == 1) {
    mapping = vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eR,
                                   vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eR);
  }

  mImageView = context->getDevice().createImageViewUnique(vk::ImageViewCreateInfo(
      {}, image.getVulkanImage(), vk::ImageViewType::e2D, image.getFormat(), mapping,
      vk::ImageSubresourceRange(getFormatAspectFlags(format), 0, 1, 0, 1)));

  mDS = ImGui_ImplVulkan_AddTexture(mSampler, mImageView.get(),
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  return std::static_pointer_cast<DisplayImage>(shared_from_this());
}

std::shared_ptr<DisplayImage> DisplayImage::Clear() {
  core::Context::Get()->getDevice().waitIdle();
  mCommandBuffer.reset();
  if (mDS) {
    ImGui_ImplVulkan_RemoveTexture(mDS);
    mDS = {};
  }

  mImageView = {};
  mSampler = vk::Sampler{};
  mLayout = {};
  mImage = {};

  return std::static_pointer_cast<DisplayImage>(shared_from_this());
}

void DisplayImage::build() {
  if (mImage) {

    // TODO: async
    core::Context::Get()->getQueue().submitAndWait(mCommandBuffer.get());

    if (mSize[0] <= 0 && mSize[1] <= 0) {
      mSize[0] = ImGui::GetWindowContentRegionWidth();
      mSize[1] = mSize[0] / mImage->getExtent().width * mImage->getExtent().height;
    }

    ImGui::Image((ImTextureID)mDS, ImVec2(mSize[0], mSize[1]));
  }
}

DisplayImage::~DisplayImage() {
  core::Context::Get()->getDevice().waitIdle();
  mCommandBuffer.reset();
  if (mDS) {
    ImGui_ImplVulkan_RemoveTexture(mDS);
    mDS = {};
  }
}

} // namespace ui
} // namespace svulkan2
