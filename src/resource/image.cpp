#include "svulkan2/resource/image.h"
#include "svulkan2/common/image.h"
#include "svulkan2/common/log.h"
#include <filesystem>

namespace fs = std::filesystem;
namespace svulkan2 {
namespace resource {

std::shared_ptr<SVImage>
SVImage::FromData(uint32_t width, uint32_t height, uint32_t channels,
                  std::vector<std::vector<uint8_t>> const &data,
                  uint32_t mipLevels) {
  if (channels != 1 && channels != 4) {
    throw std::runtime_error(
        "failed to create image: image must have 1 or 4 channels");
  }
  for (auto &d : data) {
    if (width * height * channels != d.size()) {
      throw std::runtime_error("failed to create image: image dimension does "
                               "not match image data size");
    }
  }
  auto image = std::shared_ptr<SVImage>(new SVImage);
  image->mDescription = {.source = SVImageDescription::SourceType::eCUSTOM,
                         .format = SVImageDescription::Format::eUINT8,
                         .filenames = {},
                         .mipLevels = mipLevels};
  image->mWidth = width;
  image->mHeight = height;
  image->mChannels = channels;
  image->mData = data;
  image->mLoaded = true;
  return image;
}

std::shared_ptr<SVImage>
SVImage::FromData(uint32_t width, uint32_t height, uint32_t channels,
                  std::vector<std::vector<float>> const &data,
                  uint32_t mipLevels) {
  if (channels != 1 && channels != 4) {
    throw std::runtime_error(
        "failed to create image: image must have 1 or 4 channels");
  }
  for (auto &d : data) {
    if (width * height * channels != d.size()) {
      throw std::runtime_error("failed to create image: image dimension does "
                               "not match image data size");
    }
  }
  auto image = std::shared_ptr<SVImage>(new SVImage);
  image->mDescription = {.source = SVImageDescription::SourceType::eCUSTOM,
                         .format = SVImageDescription::Format::eFLOAT,
                         .filenames = {},
                         .mipLevels = mipLevels};
  image->mWidth = width;
  image->mHeight = height;
  image->mChannels = channels;
  image->mFloatData = data;
  image->mLoaded = true;
  return image;
}

std::shared_ptr<SVImage> SVImage::FromData(uint32_t width, uint32_t height,
                                           uint32_t channels,
                                           std::vector<uint8_t> const &data,
                                           uint32_t mipLevels) {
  return SVImage::FromData(width, height, channels,
                           std::vector<std::vector<uint8_t>>{data}, mipLevels);
}

std::shared_ptr<SVImage> SVImage::FromData(uint32_t width, uint32_t height,
                                           uint32_t channels,
                                           std::vector<float> const &data,
                                           uint32_t mipLevels) {
  return SVImage::FromData(width, height, channels,
                           std::vector<std::vector<float>>{data}, mipLevels);
}

std::shared_ptr<SVImage> SVImage::FromFile(std::string const &filename,
                                           uint32_t mipLevels) {
  auto image = std::shared_ptr<SVImage>(new SVImage);
  image->mDescription =
      SVImageDescription{.source = SVImageDescription::SourceType::eFILE,
                         .filenames = {filename},
                         .mipLevels = mipLevels};
  return image;
}

std::shared_ptr<SVImage>
SVImage::FromFile(std::vector<std::string> const &filenames,
                  uint32_t mipLevels) {
  auto image = std::shared_ptr<SVImage>(new SVImage);
  image->mDescription =
      SVImageDescription{.source = SVImageDescription::SourceType::eFILE,
                         .filenames = filenames,
                         .mipLevels = mipLevels};
  return image;
}

std::shared_ptr<SVImage>
SVImage::FromDeviceImage(std::unique_ptr<core::Image> image) {
  auto result = std::shared_ptr<SVImage>(new SVImage);
  result->mDescription =
      SVImageDescription{.source = SVImageDescription::SourceType::eDEVICE};
  result->mWidth = image->getExtent().width;
  result->mHeight = image->getExtent().height;
  result->mImage = std::move(image);
  result->mLoaded = true;
  result->mOnDevice = true;
  return result;
}

void SVImage::setUsage(vk::ImageUsageFlags usage) {
  if (mOnDevice) {
    throw std::runtime_error(
        "failed to set usage: device image already created");
  }
  mUsage = usage;
}

void SVImage::setCreateFlags(vk::ImageCreateFlags flags) {
  if (mOnDevice) {
    throw std::runtime_error(
        "failed to set create flags: device image already created");
  }
  mCreateFlags = flags;
}

void SVImage::uploadToDevice(std::shared_ptr<core::Context> context,
                             bool generateMipmaps) {
  if (mOnDevice) {
    return;
  }
  mContext = context;
  if (!mLoaded) {
    throw std::runtime_error(
        "failed to upload to device: image does not exist in memory");
  }
  if (mDescription.format == SVImageDescription::Format::eUINT8) {
    mImage = std::make_unique<core::Image>(
        context, vk::Extent3D{mWidth, mHeight, 1},
        mChannels == 4 ? vk::Format::eR8G8B8A8Unorm : vk::Format::eR8Unorm,
        mUsage, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        vk::SampleCountFlagBits::e1, mDescription.mipLevels, mData.size(),
        vk::ImageTiling::eOptimal, mCreateFlags);
    for (uint32_t layer = 0; layer < mData.size(); ++layer) {
      mImage->upload(mData[layer].data(), mData[layer].size() * sizeof(uint8_t),
                     layer, generateMipmaps);
    }
  } else {
    mImage = std::make_unique<core::Image>(
        context, vk::Extent3D{mWidth, mHeight, 1},
        mChannels == 4 ? vk::Format::eR32G32B32A32Sfloat
                       : vk::Format::eR32Sfloat,
        mUsage, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        vk::SampleCountFlagBits::e1, mDescription.mipLevels, mFloatData.size(),
        vk::ImageTiling::eOptimal, mCreateFlags);
    for (uint32_t layer = 0; layer < mFloatData.size(); ++layer) {
      mImage->upload(mFloatData[layer].data(),
                     mFloatData[layer].size() * sizeof(float), layer,
                     generateMipmaps);
    }
  }
  mOnDevice = true;
}

void SVImage::removeFromDevice() {
  mOnDevice = false;
  mImage.reset();
}

std::future<void> SVImage::loadAsync() {
  if (mLoaded) {
    return std::async(std::launch::deferred, []() {});
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

    mWidth = 0;
    mHeight = 0;
    mChannels = 4;

    for (uint32_t i = 0; i < mDescription.filenames.size(); ++i) {
      int width, height;
      auto dataVector = loadImage(mDescription.filenames[i], width, height);

      if ((mWidth != 0 && mWidth != static_cast<uint32_t>(width)) ||
          (mHeight != 0 && mHeight != static_cast<uint32_t>(height))) {
        throw std::runtime_error(
            "image load failed: provided files have different sizes");
      }
      mWidth = width;
      mHeight = height;
      if (mDescription.format == SVImageDescription::Format::eUINT8) {
        mData.push_back(dataVector);
      } else {
        std::vector<float> floatDataVector;
        floatDataVector.reserve(dataVector.size());
        for (uint8_t x : dataVector) {
          floatDataVector.push_back(x);
        }
        mFloatData.push_back(floatDataVector);
      }
      mLoaded = true;
    }
  });
}

} // namespace resource
} // namespace svulkan2
