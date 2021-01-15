#pragma once
#include "image.h"
#include <memory>
#include <vector>

namespace svulkan2 {
namespace resource {

struct SVTextureDescription {
  enum SourceType { eFILE, eCUSTOM } source;
  std::string filename;
  uint32_t mipLevels;
  vk::Filter magFilter;
  vk::Filter minFilter;
  vk::SamplerAddressMode addressModeU;
  vk::SamplerAddressMode addressModeV;

  inline bool operator==(SVTextureDescription const &other) const {
    return source == other.source && filename == other.filename &&
           mipLevels == other.mipLevels && magFilter == other.magFilter &&
           minFilter == other.minFilter && addressModeU == other.addressModeU &&
           addressModeV == other.addressModeV;
  }
};

class SVTexture {
  SVTextureDescription mDescription;
  std::shared_ptr<SVImage> mImage;

  bool mOnDevice{};
  vk::UniqueImageView mImageView;
  vk::UniqueSampler mSampler;

  bool mLoaded{};

  /** When manager is not null, it is used to avoid loading duplicated
   * subresources
   */
  class SVResourceManager *mManager;

public:
  static std::shared_ptr<SVTexture> FromFile(
      std::string const &filename, uint32_t mipLevels,
      vk::Filter magFilter = vk::Filter::eLinear,
      vk::Filter minFilter = vk::Filter::eLinear,
      vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat);

  static std::shared_ptr<SVTexture> FromData(
      uint32_t width, uint32_t height, uint32_t channels,
      std::vector<uint8_t> const &data, uint32_t mipLevels = 1,
      vk::Filter magFilter = vk::Filter::eLinear,
      vk::Filter minFilter = vk::Filter::eLinear,
      vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat);

  void uploadToDevice(core::Context &context);
  void removeFromDevice();

  inline bool isOnDevice() const { return mOnDevice && mImage->isOnDevice(); }
  inline bool isLoaded() const { return mLoaded; }
  void load();

  inline SVTextureDescription const &getDescription() const {
    return mDescription;
  }

  inline void setManager(class SVResourceManager *manager) {
    mManager = manager;
  };

private:
  SVTexture();
};

} // namespace resource
} // namespace svulkan2
