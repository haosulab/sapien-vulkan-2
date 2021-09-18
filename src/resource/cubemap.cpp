#include "svulkan2/resource/cubemap.h"
#include "svulkan2/core/context.h"
#include "svulkan2/resource/manager.h"
#include "svulkan2/shader/compute.h"
#include <ktx.h>

namespace svulkan2 {
namespace resource {

std::shared_ptr<SVCubemap>
SVCubemap::FromFile(std::string const &filename, uint32_t mipLevels,
                    vk::Filter magFilter, vk::Filter minFilter, bool srgb) {
  auto texture = std::shared_ptr<SVCubemap>(new SVCubemap);
  texture->mDescription = {.source =
                               SVCubemapDescription::SourceType::eSINGLE_FILE,
                           .filenames = {filename, "", "", "", "", ""},
                           .mipLevels = mipLevels,
                           .magFilter = magFilter,
                           .minFilter = minFilter,
                           .srgb = srgb};
  return texture;
}

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
                     vk::ImageUsageFlagBits::eTransferSrc |
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

  if (mDescription.mipLevels > 1 && !mImage->mipmapIsLoaded()) {
    if (mImage->mipmapIsLoaded()) {
      log::info("Cube map mipmaps are loaded, no prefiltering required.");
    } else {
      log::info("Prefiltering cube map...");
      shader::prefilterCubemap(*mImage->getDeviceImage());
      log::info("Prefiltering cube map completed");
    }
  }
  mImage->getDeviceImage()->setCurrentLayout(
      vk::ImageLayout::eShaderReadOnlyOptimal);
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
  if (mDescription.source != SVCubemapDescription::SourceType::eFILES &&
      mDescription.source != SVCubemapDescription::SourceType::eSINGLE_FILE) {
    throw std::runtime_error(
        "failed to load texture: the texture is not specified by a file");
  }
  return std::async(std::launch::async, [this]() {
    std::lock_guard<std::mutex> lock(mLoadingMutex);
    if (mLoaded) {
      return;
    }
    if (mDescription.source == SVCubemapDescription::SourceType::eFILES) {
      auto vfiles = std::vector<std::string>(mDescription.filenames.begin(),
                                             mDescription.filenames.end());
      mImage = SVImage::FromFile(vfiles, mDescription.mipLevels);
    } else if (mDescription.source ==
               SVCubemapDescription::SourceType::eSINGLE_FILE) {
      mImage =
          SVImage::FromFile(mDescription.filenames[0], mDescription.mipLevels);
    }
    mImage->setCreateFlags(vk::ImageCreateFlagBits::eCubeCompatible);
    mImage->loadAsync().get();
    mLoaded = true;
    for (auto f : mDescription.filenames) {
      log::info("Loaded: {}", f);
    }
  });
}

void SVCubemap::load() { loadAsync().get(); }

void SVCubemap::exportKTX(std::string const &filename) {
  if (!mOnDevice) {
    throw std::runtime_error(
        "failed to export KTX, uploadToDevice should be called first.");
  }

  auto img = mImage->getDeviceImage();
  auto extent = img->getExtent();

  ktxTexture2 *texture;
  ktxTextureCreateInfo createInfo;

  createInfo.vkFormat = VK_FORMAT_R8G8B8A8_UNORM;
  createInfo.baseWidth = extent.width;
  createInfo.baseHeight = extent.height;
  createInfo.baseDepth = 1;
  createInfo.numDimensions = 2;
  createInfo.numLevels = img->getMipLevels();
  createInfo.numLayers = 1;
  createInfo.numFaces = 6;
  createInfo.isArray = KTX_FALSE;
  createInfo.generateMipmaps = KTX_FALSE;

  ktxTexture2_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture);
  std::vector<uint8_t> data(extent.width * extent.height * 4);

  for (uint32_t face = 0; face < 6; ++face) {
    uint32_t width = extent.width;
    uint32_t height = extent.height;
    for (uint32_t level = 0; level < img->getMipLevels(); ++level) {
      uint32_t size = width * height * 4;
      img->download(data.data(), size, {0, 0, 0}, {width, height, 1}, face,
                    level);
      ktxTexture_SetImageFromMemory(ktxTexture(texture), level, 0, face,
                                    data.data(), size);
      width /= 2;
      height /= 2;
      if (width == 0) {
        width = 1;
      }
      if (height == 0) {
        height = 1;
      }
    }
  }
  ktxTexture_WriteToNamedFile(ktxTexture(texture), filename.c_str());
  ktxTexture_Destroy(ktxTexture(texture));

  auto buffer = mContext->createCommandBuffer();
  buffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  img->transitionLayout(buffer.get(), vk::ImageLayout::eTransferSrcOptimal,
                        vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::AccessFlagBits::eTransferRead,
                        vk::AccessFlagBits::eShaderRead,
                        vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eFragmentShader);
  buffer->end();
  mContext->submitCommandBufferAndWait(buffer.get());
}

} // namespace resource
} // namespace svulkan2
