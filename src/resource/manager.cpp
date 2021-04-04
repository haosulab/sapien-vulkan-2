#include "svulkan2/resource/manager.h"
#include <filesystem>
#include <random>

namespace fs = std::filesystem;

namespace svulkan2 {
namespace resource {

SVResourceManager::SVResourceManager() {
  mDefaultTexture =
      SVTexture::FromData(1, 1, 4, std::vector<uint8_t>{255, 255, 255, 255});
}

std::shared_ptr<SVImage>
SVResourceManager::CreateImageFromFile(std::string const &filename,
                                       uint32_t mipLevels) {
  std::lock_guard<std::mutex> lock(mCreateLock);
  std::string path = fs::canonical(filename).string();
  SVImageDescription desc = {.source = SVImageDescription::SourceType::eFILE,
                             .filenames = {path},
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
  std::lock_guard<std::mutex> lock(mCreateLock);
  std::string path = fs::canonical(filename).string();

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

// TODO: test this function
std::shared_ptr<SVTexture>
SVResourceManager::CreateRandomTexture(std::string const &name) {
  if (mRandomTextureRegistry.find(name) != mRandomTextureRegistry.end()) {
    return mRandomTextureRegistry[name];
  }

  if (name.substr(0, 6) != "Random") {
    throw std::runtime_error("random texture name must starts with \"Random\"");
  }
  std::string n = name.substr(6);
  auto idx = n.find('x');
  if (idx == n.npos) {
    throw std::runtime_error("random texture name must starts with \"Random\"");
  }
  std::string firstNum = n.substr(0, idx);

  n = n.substr(idx + 1);
  idx = n.find('_');
  std::string secondNum = n;
  std::string seedNum = "0";
  if (idx != n.npos) {
    secondNum = n.substr(0, idx);
    seedNum = n.substr(idx + 1);
  }
  try {
    int width = std::stoi(firstNum);
    int height = std::stoi(secondNum);
    int seed = std::stoi(seedNum);
    if (width < 0 || height < 0) {
      throw std::runtime_error(
          "random texture creation failed: invalid width, height, or seed.");
    }
    auto real_rand = std::bind(std::uniform_real_distribution<float>(0, 1),
                               std::mt19937(seed));

    std::vector<float> data;
    data.reserve(width * height);
    for (int i = 0; i < width * height; ++i) {
      data.push_back(real_rand());
    }

    mRandomTextureRegistry[name] = SVTexture::FromData(width, height, 1, data);
    return mRandomTextureRegistry[name];

  } catch (std::invalid_argument const &) {
    throw std::runtime_error(
        "random texture creation failed: invalid width, height, or seed.");
  }
}

std::shared_ptr<SVModel>
SVResourceManager::CreateModelFromFile(std::string const &filename) {
  std::lock_guard<std::mutex> lock(mCreateLock);
  std::string path = fs::canonical(filename).string();

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

void SVResourceManager::setMaterialPipelineType(
    ShaderConfig::MaterialPipeline pipeline) {
  if (mMaterialPipeline == ShaderConfig::MaterialPipeline::eUNKNOWN) {
    mMaterialPipeline = pipeline;
    return;
  }
  if (mMaterialPipeline != pipeline) {
    throw std::runtime_error(
        "All shaders are required to use the same material pipeline!");
  }
}

void SVResourceManager::setVertexLayout(
    std::shared_ptr<InputDataLayout> layout) {
  if (!mVertexLayout) {
    mVertexLayout = layout;
    return;
  }
  if (*mVertexLayout != *layout) {
    throw std::runtime_error(
        "All vertex layouts are required to be the same even across renderers");
  }
}

void SVResourceManager::clearCachedResources() {
  std::lock_guard<std::mutex> lock(mCreateLock);
  mModelRegistry.clear();
  mTextureRegistry.clear();
  mImageRegistry.clear();
  mRandomTextureRegistry.clear();
}

} // namespace resource
} // namespace svulkan2
