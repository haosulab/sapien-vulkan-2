#include "svulkan2/resource/texture.h"
#include "svulkan2/core/context.h"
#include "svulkan2/resource/manager.h"

namespace svulkan2 {
namespace resource {

std::shared_ptr<SVTexture>
SVTexture::FromFile(std::string const &filename, uint32_t mipLevels,
                    vk::Filter magFilter, vk::Filter minFilter,
                    vk::SamplerAddressMode addressModeU,
                    vk::SamplerAddressMode addressModeV) {
  auto texture = std::shared_ptr<SVTexture>(new SVTexture);
  texture->mDescription = {.source = SVTextureDescription::SourceType::eFILE,
                           .filename = filename,
                           .mipLevels = mipLevels,
                           .magFilter = magFilter,
                           .minFilter = minFilter,
                           .addressModeU = addressModeU,
                           .addressModeV = addressModeV};
  return texture;
}

std::shared_ptr<SVTexture>
SVTexture::FromData(uint32_t width, uint32_t height, uint32_t channels,
                    std::vector<uint8_t> const &data, uint32_t mipLevels,
                    vk::Filter magFilter, vk::Filter minFilter,
                    vk::SamplerAddressMode addressModeU,
                    vk::SamplerAddressMode addressModeV) {
  auto texture = std::shared_ptr<SVTexture>(new SVTexture);
  texture->mDescription = {.source = SVTextureDescription::SourceType::eCUSTOM,
                           .filename = {},
                           .mipLevels = mipLevels,
                           .magFilter = magFilter,
                           .minFilter = minFilter,
                           .addressModeU = addressModeU,
                           .addressModeV = addressModeV};
  texture->mImage = SVImage::FromData(width, height, channels, data, mipLevels);
  texture->mLoaded = true;
  return texture;
}

void SVTexture::uploadToDevice(core::Context &context) {
  if (mOnDevice) {
    return;
  }
  if (!mImage->isOnDevice()) {
    mImage->uploadToDevice(context);
  }
  mImageView =
      context.getDevice().createImageViewUnique(vk::ImageViewCreateInfo(
          {}, mImage->getDeviceImage()->getVulkanImage(),
          vk::ImageViewType::e2D, mImage->getDeviceImage()->getFormat(),
          vk::ComponentSwizzle::eIdentity,
          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0,
                                    mDescription.mipLevels, 0, 1)));
  mSampler = context.getDevice().createSamplerUnique(vk::SamplerCreateInfo(
      {}, mDescription.magFilter, mDescription.minFilter,
      vk::SamplerMipmapMode::eLinear, mDescription.addressModeU,
      mDescription.addressModeV, vk::SamplerAddressMode::eRepeat, 0.f, false,
      0.f, false, vk::CompareOp::eNever, 0.f,
      static_cast<float>(mDescription.mipLevels),
      vk::BorderColor::eFloatOpaqueBlack));
  mOnDevice = true;
}

void SVTexture::removeFromDevice() {
  mOnDevice = false;
  mImageView.reset();
  mSampler.reset();
}

std::future<void> SVTexture::loadAsync() {
  if (mLoaded) {
    return std::async(std::launch::deferred, []() {});
  }
  log::info("Loading: {}", mDescription.filename);
  if (mDescription.source != SVTextureDescription::SourceType::eFILE) {
    throw std::runtime_error(
        "failed to load texture: the texture is not specified by a file");
  }
  return std::async(std::launch::async, [this]() {
    std::lock_guard<std::mutex> lock(mLoadingMutex);
    if (mLoaded) {
      return;
    }
    mImage = mManager->CreateImageFromFile(mDescription.filename,
                                           mDescription.mipLevels);
    mImage->loadAsync().get();
    mLoaded = true;
    log::info("Loaded: {}", mDescription.filename);
  });
}

void SVTexture::load() { loadAsync().get(); }

} // namespace resource
} // namespace svulkan2
