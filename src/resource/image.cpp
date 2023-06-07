#include "svulkan2/resource/image.h"
#include "../common/logger.h"
#include "svulkan2/common/image.h"
#include "svulkan2/common/launch_policy.h"
#include "svulkan2/core/context.h"
#include <filesystem>

namespace fs = std::filesystem;
namespace svulkan2 {
namespace resource {

std::shared_ptr<SVImage> SVImage::FromData(uint32_t width, uint32_t height, uint32_t channels,
                                           std::vector<std::vector<uint8_t>> const &data,
                                           uint32_t mipLevels) {
  if (channels != 1 && channels != 4) {
    throw std::runtime_error("failed to create image: image must have 1 or 4 channels");
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

std::shared_ptr<SVImage> SVImage::FromData(uint32_t width, uint32_t height, uint32_t depth,
                                           uint32_t channels,
                                           std::vector<std::vector<float>> const &data,
                                           uint32_t mipLevels) {
  if (channels != 1 && channels != 4) {
    throw std::runtime_error("failed to create image: image must have 1 or 4 channels");
  }
  for (auto &d : data) {
    if (width * height * depth * channels != d.size()) {
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
  image->mDepth = depth;
  image->mChannels = channels;
  image->mFloatData = data;
  image->mLoaded = true;
  return image;
}

std::shared_ptr<SVImage> SVImage::FromData(uint32_t width, uint32_t height, uint32_t channels,
                                           std::vector<std::vector<float>> const &data,
                                           uint32_t mipLevels) {
  return FromData(width, height, 1, channels, data, mipLevels);
}

std::shared_ptr<SVImage> SVImage::FromData(uint32_t width, uint32_t height, uint32_t channels,
                                           std::vector<uint8_t> const &data, uint32_t mipLevels) {
  return SVImage::FromData(width, height, channels, std::vector<std::vector<uint8_t>>{data},
                           mipLevels);
}

std::shared_ptr<SVImage> SVImage::FromData(uint32_t width, uint32_t height, uint32_t channels,
                                           std::vector<float> const &data, uint32_t mipLevels) {
  return SVImage::FromData(width, height, channels, std::vector<std::vector<float>>{data},
                           mipLevels);
}

std::shared_ptr<SVImage> SVImage::FromData(uint32_t width, uint32_t height, uint32_t depth,
                                           uint32_t channels, std::vector<float> const &data,
                                           uint32_t mipLevels) {
  return SVImage::FromData(width, height, depth, channels, std::vector<std::vector<float>>{data},
                           mipLevels);
}

std::shared_ptr<SVImage> SVImage::FromFile(std::string const &filename, uint32_t mipLevels) {
  auto image = std::shared_ptr<SVImage>(new SVImage);
  image->mDescription = SVImageDescription{.source = SVImageDescription::SourceType::eFILE,
                                           .filenames = {filename},
                                           .mipLevels = mipLevels};
  return image;
}

std::shared_ptr<SVImage> SVImage::FromFile(std::vector<std::string> const &filenames,
                                           uint32_t mipLevels) {
  auto image = std::shared_ptr<SVImage>(new SVImage);
  image->mDescription = SVImageDescription{.source = SVImageDescription::SourceType::eFILE,
                                           .filenames = filenames,
                                           .mipLevels = mipLevels};
  return image;
}

std::shared_ptr<SVImage> SVImage::FromDeviceImage(std::unique_ptr<core::Image> image) {
  auto result = std::shared_ptr<SVImage>(new SVImage);
  result->mDescription = SVImageDescription{.source = SVImageDescription::SourceType::eDEVICE};
  result->mWidth = image->getExtent().width;
  result->mHeight = image->getExtent().height;
  result->mImage = std::move(image);
  result->mLoaded = true;
  result->mOnDevice = true;
  return result;
}

void SVImage::setUsage(vk::ImageUsageFlags usage) {
  if (mOnDevice) {
    throw std::runtime_error("failed to set usage: device image already created");
  }
  mUsage = usage;
}

void SVImage::setCreateFlags(vk::ImageCreateFlags flags) {
  if (mOnDevice) {
    throw std::runtime_error("failed to set create flags: device image already created");
  }
  mCreateFlags = flags;
}

void SVImage::uploadToDevice(bool generateMipmaps) {
  std::scoped_lock lock(mUploadingMutex);
  if (mOnDevice) {
    return;
  }
  auto context = core::Context::Get();
  if (!mLoaded) {
    throw std::runtime_error("failed to upload to device: image does not exist in memory");
  }
  if (mDescription.format == SVImageDescription::Format::eUINT8) {
    mImage = std::make_unique<core::Image>(
        vk::Extent3D{mWidth, mHeight, mDepth},
        mChannels == 4 ? vk::Format::eR8G8B8A8Unorm : vk::Format::eR8Unorm, mUsage,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, vk::SampleCountFlagBits::e1,
        mDescription.mipLevels, mData.size(), vk::ImageTiling::eOptimal,
        mCreateFlags | vk::ImageCreateFlagBits::eMutableFormat);

    if (mMipLoaded) {
      auto pool = context->createCommandPool();
      auto cb = pool->allocateCommandBuffer();
      cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
      mImage->transitionLayout(
          cb.get(), vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, {},
          vk::AccessFlagBits::eTransferWrite, vk::PipelineStageFlagBits::eTopOfPipe,
          vk::PipelineStageFlagBits::eTransfer);
      cb->end();
      context->getQueue().submitAndWait(cb.get());

      for (uint32_t layer = 0; layer < mData.size(); ++layer) {
        uint32_t idx = 0;
        for (uint32_t l = 0; l < mDescription.mipLevels; ++l) {
          auto size = computeMipLevelSize({mWidth, mHeight, mDepth}, l) * mChannels;
          mImage->uploadLevel(mData[layer].data() + idx, size * sizeof(uint8_t), layer, l);
          idx += size;
        }
      }

      cb = pool->allocateCommandBuffer();
      cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
      mImage->transitionLayout(cb.get(), vk::ImageLayout::eTransferDstOptimal,
                               vk::ImageLayout::eShaderReadOnlyOptimal,
                               vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eFragmentShader |
                                   vk::PipelineStageFlagBits::eRayTracingShaderKHR);
      cb->end();
      context->getQueue().submitAndWait(cb.get());

    } else {
      for (uint32_t layer = 0; layer < mData.size(); ++layer) {
        mImage->upload(mData[layer].data(), mData[layer].size() * sizeof(uint8_t), layer,
                       generateMipmaps);
      }
    }
  } else {
    mImage = std::make_unique<core::Image>(
        vk::Extent3D{mWidth, mHeight, mDepth},
        mChannels == 4 ? vk::Format::eR32G32B32A32Sfloat : vk::Format::eR32Sfloat, mUsage,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, vk::SampleCountFlagBits::e1,
        mDescription.mipLevels, mFloatData.size(), vk::ImageTiling::eOptimal, mCreateFlags);
    if (mMipLoaded) {
      throw std::runtime_error("precomputed mipmaps are not supported for float textures");
    }
    for (uint32_t layer = 0; layer < mFloatData.size(); ++layer) {
      mImage->upload(mFloatData[layer].data(), mFloatData[layer].size() * sizeof(float), layer,
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
  return std::async(LAUNCH_ASYNC, [this]() {
    std::scoped_lock lock(mLoadingMutex);
    if (mLoaded) {
      return;
    }
    if (mDescription.source != SVImageDescription::SourceType::eFILE) {
      throw std::runtime_error("failed to load image: the image is not specified with a file");
    }

    mWidth = 0;
    mHeight = 0;
    mChannels = 4;

    // load ktx file
    if (mDescription.filenames.size() && mDescription.filenames[0].ends_with(".ktx")) {
      if (mDescription.format != SVImageDescription::Format::eUINT8) {
        throw std::runtime_error("Only unity is supported for ktx texture");
      }

      int width, height, levels, faces, layers;
      vk::Format format;
      auto data =
          loadKTXImage(mDescription.filenames[0], width, height, levels, faces, layers, format);
      if (format != vk::Format::eR8G8B8A8Unorm) {
        throw std::runtime_error("Only uint8 RGBA texture is currently supported for ktx");
      }

      mWidth = width;
      mHeight = height;

      size_t stride = 0;
      for (uint32_t l = 0; l < static_cast<uint32_t>(levels); ++l) {
        stride += width * height;
        width /= 2;
        height /= 2;
        if (width == 0) {
          width = 1;
        }
        if (height == 0) {
          height = 1;
        }
      }
      stride *= getFormatSize(format);

      if (mDescription.mipLevels != static_cast<uint32_t>(levels)) {
        // levels do not match, only take base level
        logger::warn("ktx texture has {} levels but {} levels requested", levels,
                     mDescription.mipLevels);
        for (uint32_t i = 0; i < static_cast<uint32_t>(faces * layers); ++i) {
          mData.push_back(std::vector<uint8_t>(
              data.begin() + i * stride,
              data.begin() + (i * stride + width * height * getFormatSize(format))));
        }
      } else {
        for (uint32_t i = 0; i < static_cast<uint32_t>(faces * layers); ++i) {
          mData.push_back(
              std::vector<uint8_t>(data.begin() + i * stride, data.begin() + ((i + 1) * stride)));
        }
        mMipLoaded = true;
      }
      mLoaded = true;
    } else {
      for (uint32_t i = 0; i < mDescription.filenames.size(); ++i) {
        int width, height;
        auto dataVector = loadImage(mDescription.filenames[i], width, height);

        if ((mWidth != 0 && mWidth != static_cast<uint32_t>(width)) ||
            (mHeight != 0 && mHeight != static_cast<uint32_t>(height))) {
          throw std::runtime_error("image load failed: provided files have different sizes");
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
    }
  });
}

} // namespace resource
} // namespace svulkan2
