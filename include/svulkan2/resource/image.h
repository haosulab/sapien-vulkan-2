#pragma once
#include "svulkan2/core/image.h"
#include <future>

namespace svulkan2 {
namespace resource {

struct SVImageDescription {
  enum class SourceType { eFILE, eCUSTOM, eDEVICE } source{SourceType::eCUSTOM};
  enum class Format { eUINT8, eFLOAT } format{Format::eUINT8};
  std::vector<std::string> filenames{};
  uint32_t mipLevels{1};

  inline bool operator==(SVImageDescription const &other) const {
    return source == other.source && filenames == other.filenames &&
           mipLevels == other.mipLevels && format == other.format;
  }
};

class SVImage {

  SVImageDescription mDescription{};
  std::unique_ptr<core::Image> mImage{};

  uint32_t mWidth{};
  uint32_t mHeight{};
  uint32_t mChannels{};
  vk::ImageCreateFlags mCreateFlags{};
  vk::ImageUsageFlags mUsage{vk::ImageUsageFlagBits::eSampled |
                             vk::ImageUsageFlagBits::eTransferDst |
                             vk::ImageUsageFlagBits::eTransferSrc};

  std::vector<std::vector<uint8_t>> mData;
  std::vector<std::vector<float>> mFloatData;
  /** the image is on the host */
  bool mLoaded{};
  bool mOnDevice{};

  /** mipmap levels are provided on host */
  bool mMipLoaded{false};

  std::mutex mLoadingMutex;
  std::mutex mUploadingMutex;

public:
  static std::shared_ptr<SVImage> FromData(uint32_t width, uint32_t height,
                                           uint32_t channels,
                                           std::vector<uint8_t> const &data,
                                           uint32_t mipLevels = 1);
  static std::shared_ptr<SVImage>
  FromData(uint32_t width, uint32_t height, uint32_t channels,
           std::vector<std::vector<uint8_t>> const &data,
           uint32_t mipLevels = 1);

  static std::shared_ptr<SVImage> FromData(uint32_t width, uint32_t height,
                                           uint32_t channels,
                                           std::vector<float> const &data,
                                           uint32_t mipLevels = 1);
  static std::shared_ptr<SVImage>
  FromData(uint32_t width, uint32_t height, uint32_t channels,
           std::vector<std::vector<float>> const &data, uint32_t mipLevels = 1);

  static std::shared_ptr<SVImage> FromFile(std::string const &filename,
                                           uint32_t mipLevels = 1);
  static std::shared_ptr<SVImage>
  FromFile(std::vector<std::string> const &filenames, uint32_t mipLevels = 1);

  static std::shared_ptr<SVImage>
  FromDeviceImage(std::unique_ptr<core::Image> image);

  void setUsage(vk::ImageUsageFlags usage);
  void setCreateFlags(vk::ImageCreateFlags flags);

  void uploadToDevice(bool generateMipmaps = true);
  void removeFromDevice();
  inline bool isLoaded() const { return mLoaded; }
  inline bool isOnDevice() const { return mOnDevice; }

  /** load the image to host memory */
  std::future<void> loadAsync();

  inline core::Image *getDeviceImage() const { return mImage.get(); }

  inline SVImageDescription const &getDescription() const {
    return mDescription;
  }

  std::vector<std::vector<uint8_t>> const &getUint8Data() const {
    return mData;
  }

  std::vector<std::vector<float>> const &getFloatData() const {
    return mFloatData;
  }

  inline uint32_t getWidth() const { return mWidth; };
  inline uint32_t getHeight() const { return mHeight; };
  inline uint32_t getChannels() const { return mChannels; };

  /** Indicate that the image loads mipmap into the data and does not require generation */
  bool mipmapIsLoaded() const { return mMipLoaded; }

private:
  SVImage() = default;
};

} // namespace resource
} // namespace svulkan2
