#include "svulkan2/resource/cubemap.h"
#include "svulkan2/core/context.h"
#include "svulkan2/resource/manager.h"
#include "svulkan2/shader/compute.h"

namespace svulkan2 {
namespace resource {

std::shared_ptr<SVCubemap>
SVCubemap::FromFile(std::array<std::string, 6> const &filenames,
                    uint32_t mipLevels, vk::Filter magFilter,
                    vk::Filter minFilter, bool srgb) {
  auto texture = std::shared_ptr<SVCubemap>(new SVCubemap);
  texture->mDescription = {.source = SVCubemapDescription::SourceType::eFILES,
                           .filenames = filenames,
                           .mipLevels = mipLevels,
                           .magFilter = magFilter,
                           .minFilter = minFilter,
                           .srgb = srgb};
  return texture;
}

std::shared_ptr<SVCubemap>
SVCubemap::FromData(uint32_t size, uint32_t channels,
                    std::array<std::vector<uint8_t>, 6> const &data,
                    uint32_t mipLevels, vk::Filter magFilter,
                    vk::Filter minFilter, bool srgb) {
  auto texture = std::shared_ptr<SVCubemap>(new SVCubemap);
  texture->mDescription = {.source = SVCubemapDescription::SourceType::eCUSTOM,
                           .filenames = {},
                           .mipLevels = mipLevels,
                           .magFilter = magFilter,
                           .minFilter = minFilter,
                           .srgb = srgb};
  std::vector<std::vector<uint8_t>> vdata(data.begin(), data.end());
  texture->mImage = SVImage::FromData(size, size, channels, vdata, mipLevels);
  texture->mLoaded = true;
  return texture;
}

void SVCubemap::uploadToDevice(std::shared_ptr<core::Context> context) {
  if (mOnDevice) {
    return;
  }
  mContext = context;
  if (!mImage->isOnDevice()) {
    mImage->setUsage(vk::ImageUsageFlagBits::eSampled |
                     vk::ImageUsageFlagBits::eTransferDst |
                     vk::ImageUsageFlagBits::eStorage);
    mImage->uploadToDevice(context, false);
  }

  auto viewFormat = mImage->getDeviceImage()->getFormat();
  if (viewFormat == vk::Format::eR8G8B8A8Unorm && mDescription.srgb) {
    viewFormat = vk::Format::eR8G8B8A8Srgb;
  }

  vk::ImageViewUsageCreateInfo usageInfo(vk::ImageUsageFlagBits::eSampled);
  vk::ImageViewCreateInfo viewInfo(
      {}, mImage->getDeviceImage()->getVulkanImage(), vk::ImageViewType::eCube,
      viewFormat, vk::ComponentSwizzle::eIdentity,
      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0,
                                mDescription.mipLevels, 0, 6));
  viewInfo.setPNext(&usageInfo);

  mImageView = context->getDevice().createImageViewUnique(viewInfo);

  mSampler = context->getDevice().createSamplerUnique(vk::SamplerCreateInfo(
      {}, mDescription.magFilter, mDescription.minFilter,
      vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, 0.f,
      false, 0.f, false, vk::CompareOp::eNever, 0.f,
      static_cast<float>(mDescription.mipLevels),
      vk::BorderColor::eFloatOpaqueBlack));

  if (mDescription.mipLevels > 1) {
    log::info("Prefiltering cube map...");
    shader::prefilterCubemap(*mImage->getDeviceImage());
    log::info("Prefiltering cube map completed");
  }

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
  if (mDescription.source != SVCubemapDescription::SourceType::eFILES) {
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
    mImage = SVImage::FromFile(vfiles, mDescription.mipLevels);
    mImage->setCreateFlags(vk::ImageCreateFlagBits::eCubeCompatible);
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
