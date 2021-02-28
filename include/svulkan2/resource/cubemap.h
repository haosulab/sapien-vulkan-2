#pragma once
#include "image.h"
#include <memory>
#include <vector>

namespace svulkan2 {
namespace resource {

struct SVCubemapDescription {
  enum SourceType { eFILE, eCUSTOM } source;
  std::array<std::string, 6> filenames;
  uint32_t mipLevels;
  vk::Filter magFilter;
  vk::Filter minFilter;
  vk::SamplerAddressMode addressModeU;
  vk::SamplerAddressMode addressModeV;

  inline bool operator==(SVCubemapDescription const &other) const {
    return source == other.source && filenames == other.filenames &&
           mipLevels == other.mipLevels && magFilter == other.magFilter &&
           minFilter == other.minFilter && addressModeU == other.addressModeU &&
           addressModeV == other.addressModeV;
  }
};

class SVCubemap {
  SVCubemapDescription mDescription;
  std::shared_ptr<SVImage> mImage;

  bool mOnDevice{};
  vk::UniqueImageView mImageView;
  vk::UniqueSampler mSampler;

  bool mLoaded{};

  class SVResourceManager *mManager;

  std::mutex mLoadingMutex;

public:
  static std::shared_ptr<SVCubemap> FromFile(
      std::array<std::string, 6> const &filenames, uint32_t mipLevels,
      vk::Filter magFilter = vk::Filter::eLinear,
      vk::Filter minFilter = vk::Filter::eLinear,
      vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat);

  static std::shared_ptr<SVCubemap> FromData(
      uint32_t size, uint32_t channels,
      std::array<std::vector<uint8_t>, 6> const &data, uint32_t mipLevels = 1,
      vk::Filter magFilter = vk::Filter::eLinear,
      vk::Filter minFilter = vk::Filter::eLinear,
      vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat);

  void uploadToDevice(core::Context &context);
  void removeFromDevice();

  inline bool isOnDevice() const { return mOnDevice && mImage->isOnDevice(); }
  inline bool isLoaded() const { return mLoaded; }
  std::future<void> loadAsync();
  void load();

  inline SVCubemapDescription const &getDescription() const {
    return mDescription;
  }

  inline void setManager(class SVResourceManager *manager) {
    mManager = manager;
  }

  inline vk::ImageView getImageView() const { return mImageView.get(); }
  inline vk::Sampler getSampler() const { return mSampler.get(); }

private:
  SVCubemap() = default;
};

} // namespace resource
} // namespace svulkan2
