#pragma once
#include "model.h"
#include "svulkan2/common/config.h"
#include <unordered_map>

namespace svulkan2 {
namespace resource {

class SVResourceManager {
  std::shared_ptr<RendererConfig> mRendererConfig;
  std::shared_ptr<ShaderConfig> mShaderConfig;

  std::unordered_map<std::string, std::vector<std::shared_ptr<SVModel>>>
      mModelRegistry;
  std::unordered_map<std::string, std::vector<std::shared_ptr<SVTexture>>>
      mTextureRegistry;
  std::unordered_map<std::string, std::vector<std::shared_ptr<SVImage>>>
      mImageRegistry;

public:
  SVResourceManager(std::shared_ptr<RendererConfig> rendererConfig,
                    std::shared_ptr<ShaderConfig> shaderConfig);

  std::shared_ptr<SVImage> CreateImageFromFile(std::string const &filename,
                                               uint32_t mipLevels);

  std::shared_ptr<SVTexture> CreateTextureFromFile(
      std::string const &filename, uint32_t mipLevels,
      vk::Filter magFilter = vk::Filter::eLinear,
      vk::Filter minFilter = vk::Filter::eLinear,
      vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat);

  std::shared_ptr<SVModel> CreateModelFromFile(std::string const &filename);

  std::shared_ptr<RendererConfig> getRendererConfig() const {
    return mRendererConfig;
  }

  std::shared_ptr<ShaderConfig> getShaderConfig() const {
    return mShaderConfig;
  }
};

} // namespace resource
} // namespace svulkan2
