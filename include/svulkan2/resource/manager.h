#pragma once
#include "cubemap.h"
#include "model.h"
#include "svulkan2/common/config.h"
#include "svulkan2/shader/shader_pack.h"
#include "svulkan2/shader/shader_pack_instance.h"
#include <memory>
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

  std::mutex mShaderPackLock{};
  std::unordered_map<std::string, std::shared_ptr<shader::ShaderPack>>
      mShaderPackRegistry;

  std::mutex mShaderPackInstanceLock{};
  std::unordered_map<std::string, std::vector<std::shared_ptr<shader::ShaderPackInstance>>>
      mShaderPackInstanceRegistry;

  std::shared_ptr<InputDataLayout> mVertexLayout{};
  std::shared_ptr<InputDataLayout> mLineVertexLayout{};

  std::shared_ptr<SVTexture> mDefaultTexture;
  std::shared_ptr<SVCubemap> mDefaultCubemap;
  std::shared_ptr<SVTexture> mDefaultBRDFLUT;

  uint32_t mDefaultMipLevels{1};

  std::mutex mCreateLock{};

public:
  SVResourceManager();
  ~SVResourceManager() = default;

  std::shared_ptr<shader::ShaderPack>
  CreateShaderPack(std::string const &dirname);

  std::shared_ptr<shader::ShaderPackInstance>
  CreateShaderPackInstance(shader::ShaderPackInstanceDesc const &desc);

  std::shared_ptr<SVImage> CreateImageFromFile(std::string const &filename,
                                               uint32_t mipLevels);

  std::shared_ptr<SVTexture> CreateTextureFromFile(
      std::string const &filename, uint32_t mipLevels,
      vk::Filter magFilter = vk::Filter::eLinear,
      vk::Filter minFilter = vk::Filter::eLinear,
      vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat,
      bool srgb = false);

  std::shared_ptr<SVCubemap>
  CreateCubemapFromKTX(std::string const &filename, uint32_t mipLevels = 1,
                       vk::Filter magFilter = vk::Filter::eLinear,
                       vk::Filter minFilter = vk::Filter::eLinear,
                       bool srgb = true);

  std::shared_ptr<SVCubemap> CreateCubemapFromFiles(
      std::array<std::string, 6> const &filenames, uint32_t mipLevels = 1,
      vk::Filter magFilter = vk::Filter::eLinear,
      vk::Filter minFilter = vk::Filter::eLinear, bool srgb = true);

  std::shared_ptr<SVTexture> generateBRDFLUT(uint32_t size);

  std::shared_ptr<SVTexture> getDefaultBRDFLUT();

  std::shared_ptr<SVTexture> CreateRandomTexture(std::string const &name);

  inline std::shared_ptr<SVTexture> getDefaultTexture() const {
    return mDefaultTexture;
  };

  inline std::shared_ptr<SVCubemap> getDefaultCubemap() const {
    return mDefaultCubemap;
  };

  std::shared_ptr<SVModel> CreateModelFromFile(std::string const &filename);

  /** release all cached resources */
  void clearCachedResources();

  /** release gpu resources.
   * NOTE: This MUST be called when no rendering is running!
   * NOTE: All renders become invalid after calling this function!*/
  void releaseGPUResourcesUnsafe();

  void setVertexLayout(std::shared_ptr<InputDataLayout> layout);
  inline std::shared_ptr<InputDataLayout> getVertexLayout() const {
    if (!mVertexLayout) {
      throw std::runtime_error(
          "[resource manager] you need to load the shader (e.g. by creating a "
          "camera) before loading objects");
    }
    return mVertexLayout;
  }

  void setLineVertexLayout(std::shared_ptr<InputDataLayout> layout);
  inline std::shared_ptr<InputDataLayout> getLineVertexLayout() const {
    if (!mLineVertexLayout) {
      throw std::runtime_error(
          "[resource manager] you need to load the shader (e.g. by creating a "
          "camera) before loading objects");
    }
    return mLineVertexLayout;
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
