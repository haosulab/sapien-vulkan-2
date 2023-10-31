#pragma once
#include "svulkan2/core/image.h"
#include <future>

namespace svulkan2 {
namespace resource {

struct SVImageDescription {
  enum class SourceType { eFILE, eCUSTOM, eDEVICE } source{SourceType::eCUSTOM};
  vk::Format format{vk::Format::eUndefined};
  std::vector<std::string> filenames{};
  uint32_t desiredChannels{0};
  uint32_t mipLevels{1};

  inline bool operator==(SVImageDescription const &other) const {
    return source == other.source && filenames == other.filenames &&
           mipLevels == other.mipLevels && format == other.format;
  }
};

class SVImage {

  SVImageDescription mDescription{};
  std::unique_ptr<core::Image> mImage{};

  vk::ImageType mType{vk::ImageType::e2D};
  vk::Format mFormat{vk::Format::eUndefined};
  uint32_t mWidth{1};
  uint32_t mHeight{1};
  uint32_t mDepth{1};
  vk::ImageCreateFlags mCreateFlags{};
  vk::ImageUsageFlags mUsage{vk::ImageUsageFlagBits::eSampled |
                             vk::ImageUsageFlagBits::eTransferDst |
                             vk::ImageUsageFlagBits::eTransferSrc};

  std::vector<std::vector<char>> mRawData;

  /** the image is on the host */
  bool mLoaded{};
  bool mOnDevice{};

  /** mipmap levels are provided on host */
  bool mMipLoaded{false};

  std::mutex mLoadingMutex;
  std::mutex mUploadingMutex;

public:
  static std::shared_ptr<SVImage> FromRawData(vk::ImageType type, uint32_t width, uint32_t height,
                                              uint32_t depth, vk::Format format,
                                              std::vector<std::vector<char>> const &data,
                                              uint32_t mipLevels = 1);

  static std::shared_ptr<SVImage> FromFile(std::vector<std::string> const &filenames,
                                           uint32_t mipLevels, uint32_t desiredChannels);

  static std::shared_ptr<SVImage> FromDeviceImage(std::unique_ptr<core::Image> image);

  void setUsage(vk::ImageUsageFlags usage);
  void setCreateFlags(vk::ImageCreateFlags flags);

  void uploadToDevice(bool generateMipmaps = true);
  void removeFromDevice();
  inline bool isLoaded() const { return mLoaded; }
  inline bool isOnDevice() const { return mOnDevice; }

  /** load the image to host memory */
  std::future<void> loadAsync();

  inline core::Image *getDeviceImage() const { return mImage.get(); }

  inline SVImageDescription const &getDescription() const { return mDescription; }

  std::vector<std::vector<char>> const &getRawData() const { return mRawData; }

  inline uint32_t getWidth() const { return mWidth; }
  inline uint32_t getHeight() const { return mHeight; }
  inline uint32_t getDepth() const { return mDepth; }
  inline uint32_t getChannels() const { return getFormatChannels(mFormat); }
  inline vk::Format getFormat() const { return mFormat; }

  /** Indicate that the image loads mipmap into the data and does not require
   * generation */
  bool mipmapIsLoaded() const { return mMipLoaded; }

private:
  SVImage() = default;
};

} // namespace resource
} // namespace svulkan2
