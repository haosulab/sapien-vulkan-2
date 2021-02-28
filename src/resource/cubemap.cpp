#include "svulkan2/resource/cubemap.h"
#include "svulkan2/core/context.h"
#include "svulkan2/resource/manager.h"

namespace svulkan2 {
namespace resource {

std::shared_ptr<SVCubemap>
SVCubemap::FromFile(std::array<std::string, 6> const &filenames,
                    uint32_t mipLevels, vk::Filter magFilter,
                    vk::Filter minFilter, vk::SamplerAddressMode addressModeU,
                    vk::SamplerAddressMode addressModeV) {
  auto texture = std::shared_ptr<SVCubemap>(new SVCubemap);
  texture->mDescription = {.source = SVCubemapDescription::SourceType::eFILE,
                           .filenames = filenames,
                           .mipLevels = mipLevels,
                           .magFilter = magFilter,
                           .minFilter = minFilter,
                           .addressModeU = addressModeU,
                           .addressModeV = addressModeV};
  return texture;
}

std::shared_ptr<SVCubemap>
SVCubemap::FromData(uint32_t size, uint32_t channels,
                    std::array<std::vector<uint8_t>, 6> const &data,
                    uint32_t mipLevels, vk::Filter magFilter,
                    vk::Filter minFilter, vk::SamplerAddressMode addressModeU,
                    vk::SamplerAddressMode addressModeV) {
  auto texture = std::shared_ptr<SVCubemap>(new SVCubemap);
  texture->mDescription = {.source = SVCubemapDescription::SourceType::eCUSTOM,
                           .filenames = {},
                           .mipLevels = mipLevels,
                           .magFilter = magFilter,
                           .minFilter = minFilter,
                           .addressModeU = addressModeU,
                           .addressModeV = addressModeV};
  std::vector<std::vector<uint8_t>> vdata(data.begin(), data.end());
  texture->mImage = SVImage::FromData(size, size, channels, vdata, mipLevels);
  texture->mLoaded = true;
  return texture;
}

void SVCubemap::uploadToDevice(core::Context &context) {
  if (mOnDevice) {
    return;
  }
  if (!mImage->isOnDevice()) {
    mImage->uploadToDevice(context);
  }
  mImageView =
      context.getDevice().createImageViewUnique(vk::ImageViewCreateInfo(
          {}, mImage->getDeviceImage()->getVulkanImage(),
          vk::ImageViewType::eCube, mImage->getDeviceImage()->getFormat(),
          vk::ComponentSwizzle::eIdentity,
          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0,
                                    mDescription.mipLevels, 0, 6)));
  mSampler = context.getDevice().createSamplerUnique(vk::SamplerCreateInfo(
      {}, mDescription.magFilter, mDescription.minFilter,
      vk::SamplerMipmapMode::eLinear, mDescription.addressModeU,
      mDescription.addressModeV, vk::SamplerAddressMode::eRepeat, 0.f, false,
      0.f, false, vk::CompareOp::eNever, 0.f,
      static_cast<float>(mDescription.mipLevels),
      vk::BorderColor::eFloatOpaqueBlack));
  mOnDevice = true;
}

void SVCubemap::removeFromDevice() {
  mOnDevice = false;
  mImageView.reset();
  mSampler.reset();
}

std::future<void> SVCubemap::loadAsync() {
  if (mLoaded) {
    return std::async(std::launch::deferred, []() {});
  }
  for (auto f : mDescription.filenames) {
    log::info("Loading: {}", f);
  }
  if (mDescription.source != SVCubemapDescription::SourceType::eFILE) {
    throw std::runtime_error(
        "failed to load texture: the texture is not specified by a file");
  }
  return std::async(std::launch::async, [this]() {
    std::lock_guard<std::mutex> lock(mLoadingMutex);
    if (mLoaded) {
      return;
    }
    auto vfiles = std::vector<std::string>(mDescription.filenames.begin(),
                                           mDescription.filenames.end());
    SVImage::FromFile(vfiles, mDescription.mipLevels);
    mImage->loadAsync().get();
    mLoaded = true;
    for (auto f : mDescription.filenames) {
      log::info("Loaded: {}", f);
    }
  });
}

void SVCubemap::load() { loadAsync().get(); }

} // namespace resource
} // namespace svulkan2
