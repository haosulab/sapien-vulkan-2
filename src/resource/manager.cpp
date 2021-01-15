#include "svulkan2/resource/manager.h"
#include <filesystem>

namespace fs = std::filesystem;

namespace svulkan2 {
namespace resource {

SVResourceManager::SVResourceManager(
    std::shared_ptr<RendererConfig> rendererConfig,
    std::shared_ptr<ShaderConfig> shaderConfig)
    : mRendererConfig(rendererConfig), mShaderConfig(shaderConfig) {}

std::shared_ptr<SVImage>
SVResourceManager::CreateImageFromFile(std::string const &filename,
                                       uint32_t mipLevels) {
  std::string path = fs::canonical(filename);
  SVImageDescription desc = {.source = SVImageDescription::SourceType::eFILE,
                             .filename = path,
                             .mipLevels = mipLevels};

  auto it = mImageRegistry.find(path);
  if (it != mImageRegistry.end()) {
    for (auto &img : it->second) {
      if (img->getDescription() == desc) {
        return img;
      }
    }
  }
  auto img = SVImage::FromFile(path, mipLevels);
  mImageRegistry[path].push_back(img);
  return img;
}

std::shared_ptr<SVTexture> SVResourceManager::CreateTextureFromFile(
    std::string const &filename, uint32_t mipLevels, vk::Filter magFilter,
    vk::Filter minFilter, vk::SamplerAddressMode addressModeU,
    vk::SamplerAddressMode addressModeV) {
  std::string path = fs::canonical(filename);

  SVTextureDescription desc = {.source =
                                   SVTextureDescription::SourceType::eFILE,
                               .filename = path,
                               .mipLevels = mipLevels,
                               .magFilter = magFilter,
                               .minFilter = minFilter,
                               .addressModeU = addressModeU,
                               .addressModeV = addressModeV};

  auto it = mTextureRegistry.find(path);
  if (it != mTextureRegistry.end()) {
    for (auto &tex : it->second) {
      if (tex->getDescription() == desc) {
        return tex;
      }
    }
  }
  auto tex = SVTexture::FromFile(path, mipLevels, magFilter, minFilter,
                                 addressModeU, addressModeV);
  tex->setManager(this);
  mTextureRegistry[path].push_back(tex);
  return tex;
}

std::shared_ptr<SVModel>
SVResourceManager::CreateModelFromFile(std::string const &filename) {
  std::string path = fs::canonical(filename);

  ModelDescription desc = {.source = ModelDescription::SourceType::eFILE,
                           .filename = filename};

  auto it = mModelRegistry.find(path);
  if (it != mModelRegistry.end()) {
    for (auto &model : it->second) {
      if (model->getDescription() == desc) {
        return model;
      }
    }
  }
  auto model = SVModel::FromFile(path);
  model->setManager(this);
  mModelRegistry[path].push_back(model);
  return model;
}

} // namespace resource
} // namespace svulkan2
