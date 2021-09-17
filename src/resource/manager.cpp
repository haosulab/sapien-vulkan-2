#include "svulkan2/resource/manager.h"
#include "svulkan2/core/context.h"
#include "svulkan2/shader/compute.h"
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
    vk::SamplerAddressMode addressModeV, bool srgb) {
  std::lock_guard<std::mutex> lock(mCreateLock);
  std::string path = fs::canonical(filename).string();

  SVTextureDescription desc = {.source =
                                   SVTextureDescription::SourceType::eFILE,
                               .filename = path,
                               .mipLevels = mipLevels,
                               .magFilter = magFilter,
                               .minFilter = minFilter,
                               .addressModeU = addressModeU,
                               .addressModeV = addressModeV,
                               .srgb = srgb};

  auto it = mTextureRegistry.find(path);
  if (it != mTextureRegistry.end()) {
    for (auto &tex : it->second) {
      if (tex->getDescription() == desc) {
        return tex;
      }
    }
  }
  auto tex = SVTexture::FromFile(path, mipLevels, magFilter, minFilter,
                                 addressModeU, addressModeV, srgb);
  tex->setManager(this);
  mTextureRegistry[path].push_back(tex);
  return tex;
}

std::shared_ptr<SVCubemap> SVResourceManager::CreateCubemapFromFiles(
    std::array<std::string, 6> const &filenames, uint32_t mipLevels,
    vk::Filter magFilter, vk::Filter minFilter) {
  std::lock_guard<std::mutex> lock(mCreateLock);
  SVCubemapDescription desc = {
      .source = SVCubemapDescription::SourceType::eFILES,
      .filenames = {fs::canonical(filenames[0]).string(),
                    fs::canonical(filenames[1]).string(),
                    fs::canonical(filenames[2]).string(),
                    fs::canonical(filenames[3]).string(),
                    fs::canonical(filenames[4]).string(),
                    fs::canonical(filenames[5]).string()},
      .mipLevels = mipLevels,
      .magFilter = magFilter,
      .minFilter = minFilter};
  auto cubemap = SVCubemap::FromFile(desc.filenames, desc.mipLevels,
                                     desc.magFilter, desc.minFilter);
  auto it = mCubemapRegistry.find(desc.filenames[0]);
  if (it != mCubemapRegistry.end()) {
    for (auto &tex : it->second) {
      if (tex->getDescription() == desc) {
        return tex;
      }
    }
  }
  cubemap->setManager(this);
  mCubemapRegistry[desc.filenames[0]].push_back(cubemap);
  return cubemap;
}

std::shared_ptr<SVTexture>
SVResourceManager::generateBRDFLUT(std::shared_ptr<core::Context> context,
                                   uint32_t size) {
  if (!context->isVulkanAvailable()) {
    return nullptr;
  }
  auto image = shader::generateBRDFLUT(context, 512);
  auto sampler = context->getDevice().createSamplerUnique(vk::SamplerCreateInfo(
      {}, vk::Filter::eLinear, vk::Filter::eLinear,
      vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge, 0.f, false, 0.f, false,
      vk::CompareOp::eNever, 0.f, 0.f, vk::BorderColor::eFloatOpaqueWhite));
  auto view =
      context->getDevice().createImageViewUnique(vk::ImageViewCreateInfo(
          {}, image->getVulkanImage(), vk::ImageViewType::e2D,
          image->getFormat(), vk::ComponentSwizzle::eIdentity,
          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0,
                                    1)));
  return resource::SVTexture::FromImage(
      resource::SVImage::FromDeviceImage(std::move(image)), std::move(view),
      std::move(sampler));
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
                           .filename = path};

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

void SVResourceManager::setLineVertexLayout(
    std::shared_ptr<InputDataLayout> layout) {
  if (!mLineVertexLayout) {
    mLineVertexLayout = layout;
    return;
  }
  if (*mLineVertexLayout != *layout) {
    throw std::runtime_error("All line vertex layouts are required to be the "
                             "same even across renderers");
  }
}

void SVResourceManager::clearCachedResources() {
  std::lock_guard<std::mutex> lock(mCreateLock);
  mModelRegistry.clear();
  mTextureRegistry.clear();
  mCubemapRegistry.clear();
  mImageRegistry.clear();
  mRandomTextureRegistry.clear();
}

} // namespace resource
} // namespace svulkan2
