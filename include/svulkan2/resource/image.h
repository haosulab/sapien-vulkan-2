#pragma once
#include "svulkan2/core/image.h"
#include <future>

namespace svulkan2 {
namespace resource {

struct SVImageDescription {
  enum SourceType { eFILE, eCUSTOM } source;
  std::vector<std::string> filenames;
  uint32_t mipLevels;

  inline bool operator==(SVImageDescription const &other) const {
    return source == other.source && filenames == other.filenames &&
           mipLevels == other.mipLevels;
  }
};

class SVImage {
  SVImageDescription mDescription{};
  std::unique_ptr<core::Image> mImage{};

  uint32_t mWidth{};
  uint32_t mHeight{};
  uint32_t mChannels{};

  std::vector<std::vector<uint8_t>> mData;
  /** the image is on the host */
  bool mLoaded{};
  bool mOnDevice{};

  std::mutex mLoadingMutex;

public:
  static std::shared_ptr<SVImage> FromData(uint32_t width, uint32_t height,
                                           uint32_t channels,
                                           std::vector<uint8_t> const &data,
                                           uint32_t mipLevels = 1);

  static std::shared_ptr<SVImage>
  FromData(uint32_t width, uint32_t height, uint32_t channels,
           std::vector<std::vector<uint8_t>> const &data,
           uint32_t mipLevels = 1);
  static std::shared_ptr<SVImage> FromFile(std::string const &filename,
                                           uint32_t mipLevels = 1);
  static std::shared_ptr<SVImage>
  FromFile(std::vector<std::string> const &filenames, uint32_t mipLevels = 1);

  void uploadToDevice(core::Context &context);
  void removeFromDevice();
  inline bool isLoaded() const { return mLoaded; }
  inline bool isOnDevice() const { return mOnDevice; }

  /** load the image to host memory */
  std::future<void> loadAsync();

  inline core::Image *getDeviceImage() const { return mImage.get(); }

  inline SVImageDescription const &getDescription() const {
    return mDescription;
  }

private:
  SVImage() = default;
};

} // namespace resource
} // namespace svulkan2
