#pragma once
#include "cubemap.h"
#include "model.h"
#include "svulkan2/common/config.h"
#include <unordered_map>

namespace svulkan2 {
namespace resource {

class SVResourceManager {
  std::unordered_map<std::string, std::vector<std::shared_ptr<SVModel>>>
      mModelRegistry;
  std::unordered_map<std::string, std::vector<std::shared_ptr<SVTexture>>>
      mTextureRegistry;
  std::unordered_map<std::string, std::vector<std::shared_ptr<SVCubemap>>>
      mCubemapRegistry;
  std::unordered_map<std::string, std::vector<std::shared_ptr<SVImage>>>
      mImageRegistry;
  std::unordered_map<std::string, std::shared_ptr<SVTexture>>
      mRandomTextureRegistry;

  std::shared_ptr<InputDataLayout> mVertexLayout{};

  std::shared_ptr<SVTexture> mDefaultTexture;

  uint32_t mDefaultMipLevels{1};

  std::mutex mCreateLock{};

public:
  SVResourceManager();
  ~SVResourceManager() = default;

  std::shared_ptr<SVImage> CreateImageFromFile(std::string const &filename,
                                               uint32_t mipLevels);

  std::shared_ptr<SVTexture> CreateTextureFromFile(
      std::string const &filename, uint32_t mipLevels,
      vk::Filter magFilter = vk::Filter::eLinear,
      vk::Filter minFilter = vk::Filter::eLinear,
      vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat);

  std::shared_ptr<SVCubemap>
  CreateCubemapFromFiles(std::array<std::string, 6> const &filenames,
                         uint32_t mipLevels = 1,
                         vk::Filter magFilter = vk::Filter::eLinear,
                         vk::Filter minFilter = vk::Filter::eLinear);

  std::shared_ptr<SVTexture>
  generateBRDFLUT(std::shared_ptr<core::Context> context, uint32_t size);

  std::shared_ptr<SVTexture> CreateRandomTexture(std::string const &name);

  inline std::shared_ptr<SVTexture> getDefaultTexture() const {
    return mDefaultTexture;
  };

  std::shared_ptr<SVModel> CreateModelFromFile(std::string const &filename);

  /** release all cached resources */
  void clearCachedResources();

  void setVertexLayout(std::shared_ptr<InputDataLayout> layout);
  inline std::shared_ptr<InputDataLayout> getVertexLayout() const {
    if (!mVertexLayout) {
      throw std::runtime_error("[resource manager] getVertexLayout called "
                               "before setVertexLayout is not allowed");
    }
    return mVertexLayout;
  }

  inline std::unordered_map<std::string,
                            std::vector<std::shared_ptr<SVModel>>> const &
  getModels() const {
    return mModelRegistry;
  }

  inline std::unordered_map<std::string,
                            std::vector<std::shared_ptr<SVTexture>>> const &
  getTextures() const {
    return mTextureRegistry;
  }

  inline std::unordered_map<std::string,
                            std::vector<std::shared_ptr<SVCubemap>>> const &
  getCubemaps() const {
    return mCubemapRegistry;
  }

  inline std::unordered_map<std::string,
                            std::vector<std::shared_ptr<SVImage>>> const &
  getImages() const {
    return mImageRegistry;
  }

  void setDefaultMipLevels(uint32_t level) { mDefaultMipLevels = level; }
  uint32_t getDefaultMipLevels() const { return mDefaultMipLevels; }
};

} // namespace resource
} // namespace svulkan2
