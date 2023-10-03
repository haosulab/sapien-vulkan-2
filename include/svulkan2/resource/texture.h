#pragma once
#include "image.h"
#include <memory>
#include <vector>

namespace svulkan2 {
namespace resource {

struct SVTextureDescription {
  enum class SourceType { eFILE, eCUSTOM } source{SourceType::eCUSTOM};

  vk::Format format{vk::Format::eUndefined};

  std::string filename{};
  uint32_t desiredChannels{0}; // affect loaded file, 0 for auto
  uint32_t mipLevels{1};
  vk::Filter magFilter{vk::Filter::eNearest};
  vk::Filter minFilter{vk::Filter::eNearest};
  vk::SamplerAddressMode addressModeU{vk::SamplerAddressMode::eRepeat};
  vk::SamplerAddressMode addressModeV{vk::SamplerAddressMode::eRepeat};
  vk::SamplerAddressMode addressModeW{vk::SamplerAddressMode::eRepeat};
  bool srgb{false};
  int dim{2};

  inline bool operator==(SVTextureDescription const &other) const {
    return source == other.source && filename == other.filename && format == other.format &&
           mipLevels == other.mipLevels && magFilter == other.magFilter &&
           minFilter == other.minFilter && addressModeU == other.addressModeU &&
           addressModeV == other.addressModeV && addressModeW == other.addressModeW &&
           srgb == other.srgb && dim == other.dim;
  }
};

class SVTexture {
public:
  static std::shared_ptr<SVTexture>
  FromFile(std::string const &filename, uint32_t mipLevels,
           vk::Filter magFilter = vk::Filter::eLinear, vk::Filter minFilter = vk::Filter::eLinear,
           vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
           vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat,
           bool srgb = false, uint32_t desiredChannels = 0);

  static std::shared_ptr<SVTexture> FromRawData(
      uint32_t width, uint32_t height, uint32_t depth, vk::Format format,
      std::vector<char> const &data, int dim, uint32_t mipLevels = 1,
      vk::Filter magFilter = vk::Filter::eLinear, vk::Filter minFilter = vk::Filter::eLinear,
      vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeW = vk::SamplerAddressMode::eRepeat, bool srgb = false);

  static std::shared_ptr<SVTexture> FromImage(std::shared_ptr<SVImage> image,
                                              vk::UniqueImageView imageView, vk::Sampler sampler);

  void uploadToDevice();
  void removeFromDevice();

  inline bool isOnDevice() const { return mOnDevice && mImage->isOnDevice(); }
  inline bool isLoaded() const { return mLoaded; }
  std::future<void> loadAsync();

  inline SVTextureDescription const &getDescription() const { return mDescription; }

  inline std::shared_ptr<SVImage> getImage() const { return mImage; }
  inline vk::ImageView getImageView() const { return mImageView.get(); }
  inline vk::Sampler getSampler() const { return mSampler; }

  SVTexture(SVTexture const &other) = delete;
  SVTexture &operator=(SVTexture const &other) = delete;
  SVTexture(SVTexture &&other) = delete;
  SVTexture &operator=(SVTexture &&other) = delete;
  inline SVTexture() {}

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
};

} // namespace resource
} // namespace svulkan2
