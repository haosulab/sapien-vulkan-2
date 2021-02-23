#include "svulkan2/resource/image.h"
#include "svulkan2/common/log.h"
#include <filesystem>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include <stb_image.h>
#pragma GCC diagnostic pop

namespace fs = std::filesystem;
namespace svulkan2 {
namespace resource {

std::shared_ptr<SVImage> SVImage::FromData(uint32_t width, uint32_t height,
                                           uint32_t channels,
                                           std::vector<uint8_t> const &data,
                                           uint32_t mipLevels) {
  if (channels != 1 && channels != 4) {
    throw std::runtime_error(
        "failed to create image: image must have 1 or 4 channels");
  }
  if (width * height * channels != data.size()) {
    throw std::runtime_error("failed to create image: image dimension does "
                             "not match image data size");
  }
  auto image = std::shared_ptr<SVImage>(new SVImage);
  image->mDescription = {.source = SVImageDescription::SourceType::eCUSTOM,
                         .filename = {},
                         .mipLevels = mipLevels};
  image->mWidth = width;
  image->mHeight = height;
  image->mChannels = channels;
  image->mData = data;
  image->mLoaded = true;
  return image;
}

std::shared_ptr<SVImage> SVImage::FromFile(std::string const &filename,
                                           uint32_t mipLevels) {
  auto image = std::shared_ptr<SVImage>(new SVImage);
  image->mDescription = {.source = SVImageDescription::SourceType::eFILE,
                         .filename = filename,
                         .mipLevels = mipLevels};
  return image;
}

void SVImage::uploadToDevice(core::Context &context) {
  if (mOnDevice) {
    return;
  }
  if (!mLoaded) {
    throw std::runtime_error(
        "failed to upload to device: image does not exist in memory");
  }
  mImage = std::make_unique<core::Image>(
      context, vk::Extent3D{mWidth, mHeight, 1},
      mChannels == 4 ? vk::Format::eR8G8B8A8Unorm : vk::Format::eR8Unorm,
      vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
          vk::ImageUsageFlagBits::eTransferSrc,
      VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, vk::SampleCountFlagBits::e1,
      mDescription.mipLevels);
  mImage->upload(mData.data(), mData.size() * sizeof(uint8_t));
  mOnDevice = true;
}

void SVImage::removeFromDevice() {
  mOnDevice = false;
  mImage.reset();
}

std::future<void> SVImage::loadAsync() {
  if (mLoaded) {
    return std::async(std::launch::deferred, [](){});
  }
  return std::async(std::launch::async, [this]() {
    std::lock_guard<std::mutex> lock(mLoadingMutex);
    if (mLoaded) {
      return;
    }
    if (mDescription.source != SVImageDescription::SourceType::eFILE) {
      throw std::runtime_error(
          "failed to load image: the image is not specified with a file");
    }

    int width, height, nrChannels;
    unsigned char *data = stbi_load(mDescription.filename.c_str(), &width,
                                    &height, &nrChannels, STBI_rgb_alpha);
    mWidth = width;
    mHeight = height;
    mChannels = 4;
    mData = std::vector(data, data + width * height * 4);
    stbi_image_free(data);
    mLoaded = true;
  });
}

} // namespace resource
} // namespace svulkan2
