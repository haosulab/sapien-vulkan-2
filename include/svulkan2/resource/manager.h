#pragma once
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
  std::unordered_map<std::string, std::vector<std::shared_ptr<SVImage>>>
      mImageRegistry;

  ShaderConfig::MaterialPipeline mMaterialPipeline{
      ShaderConfig::MaterialPipeline::eUNKNOWN};
  std::shared_ptr<InputDataLayout> mVertexLayout{};

  std::shared_ptr<SVTexture> mDefaultTexture;

public:
  SVResourceManager();

  std::shared_ptr<SVImage> CreateImageFromFile(std::string const &filename,
                                               uint32_t mipLevels);

  std::shared_ptr<SVTexture> CreateTextureFromFile(
      std::string const &filename, uint32_t mipLevels,
      vk::Filter magFilter = vk::Filter::eLinear,
      vk::Filter minFilter = vk::Filter::eLinear,
      vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat);

  inline std::shared_ptr<SVTexture> getDefaultTexture() const {
    return mDefaultTexture;
  };

  std::shared_ptr<SVModel> CreateModelFromFile(std::string const &filename);

  void setMaterialPipelineType(ShaderConfig::MaterialPipeline pipeline);
  inline ShaderConfig::MaterialPipeline getMaterialPipelineType() const {
    return mMaterialPipeline;
  }

  void setVertexLayout(std::shared_ptr<InputDataLayout> layout);
  inline std::shared_ptr<InputDataLayout> getVertexLayout() const {
    return mVertexLayout;
  }
};

} // namespace resource
} // namespace svulkan2
