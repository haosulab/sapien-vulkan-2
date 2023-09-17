#include "svulkan2/resource/texture.h"
#include "../common/logger.h"
#include "svulkan2/common/launch_policy.h"
#include "svulkan2/core/context.h"
#include "svulkan2/resource/manager.h"

namespace svulkan2 {
namespace resource {

std::shared_ptr<SVTexture> SVTexture::FromFile(std::string const &filename, uint32_t mipLevels,
                                               vk::Filter magFilter, vk::Filter minFilter,
                                               vk::SamplerAddressMode addressModeU,
                                               vk::SamplerAddressMode addressModeV, bool srgb) {
  auto texture = std::shared_ptr<SVTexture>(new SVTexture);
  texture->mDescription = SVTextureDescription{.source = SVTextureDescription::SourceType::eFILE,
                                               .filename = filename,
                                               .mipLevels = mipLevels,
                                               .magFilter = magFilter,
                                               .minFilter = minFilter,
                                               .addressModeU = addressModeU,
                                               .addressModeV = addressModeV,
                                               .srgb = srgb,
                                               .dim = 2};
  return texture;
}

std::shared_ptr<SVTexture> SVTexture::FromRawData(uint32_t width, uint32_t height, uint32_t depth,
                                                  vk::Format format, std::vector<char> const &data,
                                                  int dim, uint32_t mipLevels,
                                                  vk::Filter magFilter, vk::Filter minFilter,
                                                  vk::SamplerAddressMode addressModeU,
                                                  vk::SamplerAddressMode addressModeV,
                                                  vk::SamplerAddressMode addressModeW, bool srgb) {
  if (srgb && !getFormatSupportSrgb(format)) {
    throw std::runtime_error("format does not support srgb");
  }

  auto texture = std::shared_ptr<SVTexture>(new SVTexture);
  texture->mDescription = {.source = SVTextureDescription::SourceType::eCUSTOM,
                           .format = format,
                           .filename = {},
                           .mipLevels = mipLevels,
                           .magFilter = magFilter,
                           .minFilter = minFilter,
                           .addressModeU = addressModeU,
                           .addressModeV = addressModeV,
                           .addressModeW = addressModeW,
                           .srgb = srgb,
                           .dim = dim};
  if (dim <= 0 || dim > 3) {
    throw std::runtime_error("texture dimension must be 1, 2 or 3");
  }
  if (dim == 1 && (height != 1 || depth != 1)) {
    throw std::runtime_error("1D texture must have height = 1 and depth = 1");
  }
  if (dim == 2 && (depth != 1)) {
    throw std::runtime_error("2D texture must have depth = 1");
  }

  texture->mImage = SVImage::FromRawData(width, height, depth, format, {data}, mipLevels);
  texture->mLoaded = true;
  return texture;
}

// std::shared_ptr<SVTexture> SVTexture::FromData(uint32_t width, uint32_t height, uint32_t
// channels,
//                                                std::vector<uint8_t> const &data,
//                                                uint32_t mipLevels, vk::Filter magFilter,
//                                                vk::Filter minFilter,
//                                                vk::SamplerAddressMode addressModeU,
//                                                vk::SamplerAddressMode addressModeV, bool srgb) {
//   auto texture = std::shared_ptr<SVTexture>(new SVTexture);
//   texture->mDescription = SVTextureDescription{.source =
//   SVTextureDescription::SourceType::eCUSTOM,
//                                                .format = SVTextureDescription::Format::eUINT8,
//                                                .filename = {},
//                                                .mipLevels = mipLevels,
//                                                .magFilter = magFilter,
//                                                .minFilter = minFilter,
//                                                .addressModeU = addressModeU,
//                                                .addressModeV = addressModeV,
//                                                .srgb = srgb};
//   texture->mImage = SVImage::FromData(width, height, channels, data, mipLevels);
//   texture->mLoaded = true;
//   return texture;
// }

// std::shared_ptr<SVTexture> SVTexture::FromData(uint32_t width, uint32_t height, uint32_t
// channels,
//                                                std::vector<float> const &data, uint32_t
//                                                mipLevels, vk::Filter magFilter, vk::Filter
//                                                minFilter, vk::SamplerAddressMode addressModeU,
//                                                vk::SamplerAddressMode addressModeV) {
//   auto texture = std::shared_ptr<SVTexture>(new SVTexture);
//   texture->mDescription = {.source = SVTextureDescription::SourceType::eCUSTOM,
//                            .format = SVTextureDescription::Format::eFLOAT,
//                            .filename = {},
//                            .mipLevels = mipLevels,
//                            .magFilter = magFilter,
//                            .minFilter = minFilter,
//                            .addressModeU = addressModeU,
//                            .addressModeV = addressModeV};
//   texture->mImage = SVImage::FromData(width, height, channels, data, mipLevels);
//   texture->mLoaded = true;
//   return texture;
// }

// std::shared_ptr<SVTexture> SVTexture::FromData(uint32_t width, uint32_t height, uint32_t depth,
//                                                uint32_t channels, std::vector<float> const
//                                                &data, int dim, uint32_t mipLevels, vk::Filter
//                                                magFilter, vk::Filter minFilter,
//                                                vk::SamplerAddressMode addressModeU,
//                                                vk::SamplerAddressMode addressModeV,
//                                                vk::SamplerAddressMode addressModeW) {
//   auto texture = std::shared_ptr<SVTexture>(new SVTexture);
//   texture->mDescription = {.source = SVTextureDescription::SourceType::eCUSTOM,
//                            .format = SVTextureDescription::Format::eFLOAT,
//                            .filename = {},
//                            .mipLevels = mipLevels,
//                            .magFilter = magFilter,
//                            .minFilter = minFilter,
//                            .addressModeU = addressModeU,
//                            .addressModeV = addressModeV,
//                            .addressModeW = addressModeW,
//                            .dim = dim};
//   if (dim <= 0 || dim > 3) {
//     throw std::runtime_error("texture dimension must be 1, 2 or 3");
//   }
//   if (dim == 1 && (height != 1 || depth != 1)) {
//     throw std::runtime_error("1D texture must have height = 1 and depth = 1");
//   }
//   if (dim == 2 && (depth != 1)) {
//     throw std::runtime_error("2D texture must have depth = 1");
//   }

//   texture->mImage = SVImage::FromData(width, height, depth, channels, data, mipLevels);
//   texture->mLoaded = true;
//   return texture;
// }

std::shared_ptr<SVTexture> SVTexture::FromImage(std::shared_ptr<SVImage> image,
                                                vk::UniqueImageView imageView,
                                                vk::Sampler sampler) {
  auto texture = std::shared_ptr<SVTexture>(new SVTexture);
  texture->mContext = core::Context::Get();
  texture->mDescription = {.source = SVTextureDescription::SourceType::eCUSTOM};
  texture->mImage = image;
  texture->mImageView = std::move(imageView);
  texture->mSampler = sampler;
  texture->mLoaded = true;
  return texture;
}

void SVTexture::uploadToDevice() {
  std::scoped_lock lock(mUploadingMutex);
  if (mOnDevice) {
    return;
  }
  mContext = core::Context::Get();

  if (!mImage->isOnDevice()) {
    mImage->uploadToDevice();
  }
  if (!mImageView) {
    auto format = mImage->getDeviceImage()->getFormat();
    if (mDescription.srgb) {
      if (format == vk::Format::eR8G8B8A8Unorm) {
        format = vk::Format::eR8G8B8A8Srgb;
      } else if (format == vk::Format::eR8Unorm) {
        format = vk::Format::eR8Srgb;
      }
    }

    vk::ImageViewType viewType = vk::ImageViewType::e2D;
    if (mDescription.dim == 1) {
      viewType = vk::ImageViewType::e1D;
    } else if (mDescription.dim == 2) {
      viewType = vk::ImageViewType::e2D;
    } else {
      viewType = vk::ImageViewType::e3D;
    }

    mImageView = mContext->getDevice().createImageViewUnique(
        vk::ImageViewCreateInfo({}, mImage->getDeviceImage()->getVulkanImage(), viewType, format,
                                vk::ComponentSwizzle::eIdentity,
                                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0,
                                                          mDescription.mipLevels, 0, 1)));
  }
  if (!mSampler) {
    mSampler = mContext->createSampler(vk::SamplerCreateInfo(
        {}, mDescription.magFilter, mDescription.minFilter, vk::SamplerMipmapMode::eLinear,
        mDescription.addressModeU, mDescription.addressModeV, mDescription.addressModeW, 0.f,
        false, 0.f, false, vk::CompareOp::eNever, 0.f, static_cast<float>(mDescription.mipLevels),
        vk::BorderColor::eFloatOpaqueBlack));
  }
  mOnDevice = true;
}

void SVTexture::removeFromDevice() {
  mOnDevice = false;
  mImageView.reset();
}

std::future<void> SVTexture::loadAsync() {
  if (mLoaded) {
    return std::async(std::launch::deferred, []() {});
  }

  auto context = core::Context::Get();
  auto manager = context->getResourceManager();

  logger::info("Loading: {}", mDescription.filename);
  if (mDescription.source != SVTextureDescription::SourceType::eFILE) {
    throw std::runtime_error("failed to load texture: the texture is not specified by a file");
  }
  return std::async(LAUNCH_ASYNC, [this, manager]() {
    std::scoped_lock lock(mLoadingMutex);
    if (mLoaded) {
      return;
    }
    mImage = manager->CreateImageFromFile(mDescription.filename, mDescription.mipLevels);
    mImage->loadAsync().get();
    mLoaded = true;
    logger::info("Loaded: {}", mDescription.filename);
  });
}

} // namespace resource
} // namespace svulkan2
