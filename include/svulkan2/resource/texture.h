#pragma once
#include "image.h"
#include <memory>
#include <vector>

namespace svulkan2 {
namespace resource {

struct SVTextureDescription {
  enum class SourceType { eFILE, eCUSTOM } source{SourceType::eCUSTOM};
  enum class Format { eUINT8, eFLOAT, eUNKNOWN } format{Format::eUINT8};
  std::string filename{};
  uint32_t mipLevels{1};
  vk::Filter magFilter{vk::Filter::eNearest};
  vk::Filter minFilter{vk::Filter::eNearest};
  vk::SamplerAddressMode addressModeU{vk::SamplerAddressMode::eRepeat};
  vk::SamplerAddressMode addressModeV{vk::SamplerAddressMode::eRepeat};
  bool srgb{false};

  inline bool operator==(SVTextureDescription const &other) const {
    return source == other.source && filename == other.filename &&
           format == other.format && mipLevels == other.mipLevels &&
           magFilter == other.magFilter && minFilter == other.minFilter &&
           addressModeU == other.addressModeU &&
           addressModeV == other.addressModeV && srgb == other.srgb;
  }
};

class SVTexture {
public:
  static std::shared_ptr<SVTexture> FromFile(
      std::string const &filename, uint32_t mipLevels,
      vk::Filter magFilter = vk::Filter::eLinear,
      vk::Filter minFilter = vk::Filter::eLinear,
      vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat,
      bool srgb = false);

  static std::shared_ptr<SVTexture> FromData(
      uint32_t width, uint32_t height, uint32_t channels,
      std::vector<uint8_t> const &data, uint32_t mipLevels = 1,
      vk::Filter magFilter = vk::Filter::eLinear,
      vk::Filter minFilter = vk::Filter::eLinear,
      vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat,
      bool srgb = false);

  static std::shared_ptr<SVTexture> FromData(
      uint32_t width, uint32_t height, uint32_t channels,
      std::vector<float> const &data, uint32_t mipLevels = 1,
      vk::Filter magFilter = vk::Filter::eLinear,
      vk::Filter minFilter = vk::Filter::eLinear,
      vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat);

  static std::shared_ptr<SVTexture> FromImage(std::shared_ptr<SVImage> image,
                                              vk::UniqueImageView imageView,
                                              vk::Sampler sampler);

  void uploadToDevice();
  void removeFromDevice();

  inline bool isOnDevice() const { return mOnDevice && mImage->isOnDevice(); }
  inline bool isLoaded() const { return mLoaded; }
  std::future<void> loadAsync();

  inline SVTextureDescription const &getDescription() const {
    return mDescription;
  }

  inline std::shared_ptr<SVImage> getImage() const { return mImage; }
  inline vk::ImageView getImageView() const { return mImageView.get(); }
  inline vk::Sampler getSampler() const { return mSampler; }

  SVTexture(SVTexture const &other) = delete;
  SVTexture &operator=(SVTexture const &other) = delete;
  SVTexture(SVTexture &&other) = delete;
  SVTexture &operator=(SVTexture &&other) = delete;
  inline SVTexture() {}

  inline int getGlobalIndex() const { return mGlobalIndex; }
  inline void setGlobalIndex(int index) { mGlobalIndex = index; }

private:
  std::shared_ptr<core::Context> mContext; // keep alive for sampler and
                                           // image view
  SVTextureDescription mDescription{};
  std::shared_ptr<SVImage> mImage{};

  bool mOnDevice{};
  vk::UniqueImageView mImageView{};
  vk::Sampler mSampler{};

  bool mLoaded{};

  std::mutex mUploadingMutex;
  std::mutex mLoadingMutex;

  int mGlobalIndex{-1}; // global index global array
};

} // namespace resource
} // namespace svulkan2
