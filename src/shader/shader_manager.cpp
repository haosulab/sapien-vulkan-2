#include "svulkan2/shader/shader_manager.h"
#include "svulkan2/common/err.h"
#include <filesystem>
#include <set>

namespace fs = std::filesystem;

namespace svulkan2 {
namespace shader {

inline static std::string
getInTextureName(std::string variableName) { // remove "sampler" prefix
  return variableName.substr(7, std::string::npos);
}

ShaderManager::ShaderManager(std::shared_ptr<RendererConfig> config)
    : mRenderConfig(config) {
  mGbufferPass = std::make_shared<GbufferPassParser>();
  mDeferredPass = std::make_shared<DeferredPassParser>();
  mShaderConfig = std::make_shared<ShaderConfig>();
  processShadersInFolder(config->shaderDir);
}

void ShaderManager::processShadersInFolder(std::string const &path) {
  mNumPasses = 0;
  if (!fs::is_directory(fs::path(path))) {
    throw std::runtime_error("[shader manager] " + path +
                             " is not a directory");
  }
  if (!fs::is_regular_file(fs::path(path) / "gbuffer.vert")) {
    throw std::runtime_error("[shader manager] gbuffer.vert is required");
  }
  if (!fs::is_regular_file(fs::path(path) / "gbuffer.frag")) {
    throw std::runtime_error("[shader manager] gbuffer.frag is required");
  }
  if (!fs::is_regular_file(fs::path(path) / "deferred.vert")) {
    throw std::runtime_error("[shader manager] deferred.vert is required");
  }
  if (!fs::is_regular_file(fs::path(path) / "deferred.frag")) {
    throw std::runtime_error("[shader manager] deferred.frag is required");
  }

  // load gbuffer pass
  std::string vsFile = path + "/gbuffer.vert";
  std::string fsFile = path + "/gbuffer.frag";
  mGbufferPass->loadGLSLFiles(vsFile, fsFile);
  mPassIndex[mGbufferPass] = mNumPasses++;

  // load deferred pass
  vsFile = path + "/deferred.vert";
  fsFile = path + "/deferred.frag";
  mDeferredPass->loadGLSLFiles(vsFile, fsFile);
  mPassIndex[mDeferredPass] = mNumPasses++;

  int numCompositePasses = 0;
  for (const auto &entry : fs::directory_iterator(path)) {
    std::string filename = entry.path().filename().string();
    if (filename.substr(0, 9) == "composite" &&
        filename.substr(filename.length() - 5, 5) == ".frag")
      numCompositePasses++;
    // FIXME: do an error check here (composite passes must be consecutive integers starting from 0)
  }

  mCompositePasses.resize(numCompositePasses);
  vsFile = path + "/composite.vert";
  for (int i = 0; i < numCompositePasses; i++) {
    fsFile = path + "/composite" + std::to_string(i) + ".frag";
    mCompositePasses[i] = std::make_shared<CompositePassParser>();
    mCompositePasses[i]->loadGLSLFiles(vsFile, fsFile);
    mPassIndex[mCompositePasses[i]] = mNumPasses++;
  }
  populateShaderConfig();
  prepareRenderTargetFormats();
  prepareRenderTargetOperationTable();
}

void ShaderManager::populateShaderConfig() {
  mShaderConfig->materialPipeline = mGbufferPass->getMaterialType();
  mShaderConfig->vertexLayout = mGbufferPass->getVertexInputLayout();
  mShaderConfig->objectBufferLayout = mGbufferPass->getObjectBufferLayout();
  mShaderConfig->sceneBufferLayout = mDeferredPass->getSceneBufferLayout();
  mShaderConfig->cameraBufferLayout = mGbufferPass->getCameraBufferLayout();
}

void ShaderManager::prepareRenderTargetFormats() {
  auto allPasses = getAllPasses();
  mRenderTargetFormats["Depth"] = mRenderConfig->depthFormat;

  for (auto pass : allPasses) {
    for (auto &elem : pass->getTextureOutputLayout()->elements) {
      std::string texName = getOutTextureName(elem.second.name);
      if (texName == "Depth") {
        throw std::runtime_error(
            "You are not allowed to name your texture \"Depth\"");
      }
      vk::Format texFormat;
      switch (elem.second.dtype) {
      case eFLOAT4:
        texFormat = mRenderConfig->colorFormat;
        break;
      case eUINT4:
        texFormat = vk::Format::eR32G32B32A32Uint;
        break;
      default:
        throw std::runtime_error("Unsupported texture format");
      }
      if (mRenderTargetFormats.find(texName) != mRenderTargetFormats.end()) {
        if (mRenderTargetFormats[texName] != texFormat) {
          std::runtime_error("Inconsistent texture format for \"" + texName +
                             "\"");
        }
      }
      mRenderTargetFormats[texName] = texFormat;
    }
  }
}

void ShaderManager::prepareRenderTargetOperationTable() {
  mTextureOperationTable = {};
  for (auto tex : mRenderTargetFormats) {
    mTextureOperationTable[tex.first] = std::vector<TextureOperation>(
        mNumPasses, TextureOperation::eTextureNoOp);
  }

  // process gbuffer out textures:
  for (auto &elem : mGbufferPass->getTextureOutputLayout()->elements) {
    std::string texName = getOutTextureName(elem.second.name);
    mTextureOperationTable[texName][mPassIndex[mGbufferPass]] =
        TextureOperation::eTextureWrite;
  }

  // process input textures of deferred pass:
  for (auto &elem : mDeferredPass->getCombinedSamplerLayout()->elements) {
    std::string texName = getInTextureName(elem.second.name);
    if (mTextureOperationTable.find(texName) != mTextureOperationTable.end()) {
      mTextureOperationTable[texName][mPassIndex[mDeferredPass]] =
          TextureOperation::eTextureRead;
    }
  }

  // process out textures of deferred paas:
  for (auto &elem : mDeferredPass->getTextureOutputLayout()->elements) {
    std::string texName = getOutTextureName(elem.second.name);
    mTextureOperationTable[texName][mPassIndex[mDeferredPass]] =
        TextureOperation::eTextureWrite;
  }

  for (uint32_t i = 0; i < mCompositePasses.size(); i++) {
    auto compositePass = mCompositePasses[i];
    // process input textures of composite pass:
    for (auto &elem : compositePass->getCombinedSamplerLayout()->elements) {
      std::string texName = getInTextureName(elem.second.name);
      if (mTextureOperationTable.find(texName) !=
          mTextureOperationTable.end()) {
        mTextureOperationTable[texName][mPassIndex[mCompositePasses[i]]] =
            TextureOperation::eTextureRead;
      }
    }
    // add composite out texture to the set:
    for (auto &elem : compositePass->getTextureOutputLayout()->elements) {
      std::string texName = getOutTextureName(elem.second.name);
      mTextureOperationTable[texName][mPassIndex[mCompositePasses[i]]] =
          TextureOperation::eTextureWrite;
    }
  }
}

TextureOperation
ShaderManager::getNextOperation(std::string texName,
                                std::shared_ptr<BaseParser> pass) {
  for (uint32_t i = mPassIndex[pass] + 1; i < mNumPasses; i++) {
    TextureOperation op = mTextureOperationTable[texName][i];
    if (op != TextureOperation::eTextureNoOp) {
      return op;
    }
  }
  return TextureOperation::eTextureNoOp;
}

TextureOperation
ShaderManager::getPrevOperation(std::string texName,
                                std::shared_ptr<BaseParser> pass) {
  for (int i = mPassIndex[pass] - 1; i >= 0; i--) {
    TextureOperation op = mTextureOperationTable[texName][i];
    if (op != TextureOperation::eTextureNoOp) {
      return op;
    }
  }
  return TextureOperation::eTextureNoOp;
}

std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>>
ShaderManager::getColorRenderTargetLayoutsForPass(
    std::shared_ptr<BaseParser> pass) {
  auto elems = pass->getTextureOutputLayout()->getElementsSorted();
  std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> result(elems.size());
  for (uint32_t i = 0; i < elems.size(); ++i) {
    ASSERT(elems[i].location == i,
           "output textures must have consecutive binding locations");
    auto texName = getOutTextureName(elems[i].name);
    auto prevOp = getPrevOperation(texName, pass);
    auto nextOp = getNextOperation(texName, pass);
    if (prevOp == TextureOperation::eTextureNoOp) {
      result[i].first = vk::ImageLayout::eUndefined;
    } else {
      result[i].first = vk::ImageLayout::eColorAttachmentOptimal;
    }
    switch (nextOp) {
    case TextureOperation::eTextureNoOp:
      result[i].second = vk::ImageLayout::eTransferSrcOptimal;
      break;
    case TextureOperation::eTextureRead:
      result[i].second = vk::ImageLayout::eShaderReadOnlyOptimal;
      break;
    case TextureOperation::eTextureWrite:
      result[i].second = vk::ImageLayout::eColorAttachmentOptimal;
      break;
    }
  }
  return result;
}

// std::unordered_map<std::string, std::pair<vk::ImageLayout, vk::ImageLayout>>
// ShaderManager::getRenderTargetLayouts(
//     std::shared_ptr<BaseParser> pass,
//     std::shared_ptr<OutputDataLayout> outputLayout) {
//   std::unordered_map<std::string, std::pair<vk::ImageLayout,
//   vk::ImageLayout>>
//       layouts;
//   for (auto &elem : outputLayout->elements) {
//     std::string texName = getOutTextureName(elem.second.name);

//     // compute initial layout:
//     TextureOperation prevOp = getPrevOperation(texName, pass);
//     if (prevOp == TextureOperation::eTextureNoOp) {
//       layouts[texName].first = vk::ImageLayout::eUndefined;
//     } else { // this layout would have been set by prev paas
//       layouts[texName].first = vk::ImageLayout::eColorAttachmentOptimal;
//     }

//     // compute final layout:
//     TextureOperation nextOp = getNextOperation(texName, pass);
//     switch (nextOp) {
//     case TextureOperation::eTextureNoOp:
//       layouts[texName].second = vk::ImageLayout::eTransferSrcOptimal;
//       break;
//     case TextureOperation::eTextureRead:
//       layouts[texName].second = vk::ImageLayout::eShaderReadOnlyOptimal;
//       break;
//     case TextureOperation::eTextureWrite:
//       layouts[texName].second = vk::ImageLayout::eColorAttachmentOptimal;
//       break;
//     default:
//       break;
//     }
//   }
//   return layouts;
// }

void ShaderManager::createDescriptorSetLayouts(vk::Device device) {
  // TODO: read shader and create layouts for the other shaders
  {
    vk::DescriptorSetLayoutBinding binding(
        0, vk::DescriptorType::eUniformBuffer, 1,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
    mObjectLayout = device.createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo({}, 1, &binding));
  }
  {
    vk::DescriptorSetLayoutBinding binding(
        0, vk::DescriptorType::eUniformBuffer, 1,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
    mCameraLayout = device.createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo({}, 1, &binding));
  }
  {
    vk::DescriptorSetLayoutBinding binding(
        0, vk::DescriptorType::eUniformBuffer, 1,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
    mSceneLayout = device.createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo({}, 1, &binding));
  }
  {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    auto textures =
        mDeferredPass->getCombinedSamplerLayout()->getElementsSorted();
    for (auto tex : textures) {
      bindings.push_back(vk::DescriptorSetLayoutBinding(
          tex.binding, vk::DescriptorType::eCombinedImageSampler, 1,
          vk::ShaderStageFlagBits::eFragment));
    }
    mDeferredLayout = device.createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo({}, bindings.size(),
                                          bindings.data()));
  }
  for (auto pass : mCompositePasses) {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    auto textures = pass->getCombinedSamplerLayout()->getElementsSorted();
    for (auto tex : textures) {
      bindings.push_back(vk::DescriptorSetLayoutBinding(
          tex.binding, vk::DescriptorType::eCombinedImageSampler, 1,
          vk::ShaderStageFlagBits::eFragment));
    }
    mCompositeLayouts.push_back(device.createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo({}, bindings.size(),
                                          bindings.data())));
  }
}

void ShaderManager::createPipelines(core::Context &context,
                                    int numDirectionalLights,
                                    int numPointLights) {
  auto device = context.getDevice();
  createDescriptorSetLayouts(device);
  mGbufferPass->createGraphicsPipeline(
      device, mRenderConfig->colorFormat, mRenderConfig->depthFormat,
      mRenderConfig->culling, vk::FrontFace::eCounterClockwise,
      getColorRenderTargetLayoutsForPass(mGbufferPass),
      {mCameraLayout.get(), mObjectLayout.get(),
       mGbufferPass->getMaterialType() ==
               ShaderConfig::MaterialPipeline::eMETALLIC
           ? context.getMetallicDescriptorSetLayout()
           : context.getSpecularDescriptorSetLayout()});

  mDeferredPass->createGraphicsPipeline(
      device, mRenderConfig->colorFormat, mRenderConfig->depthFormat,
      getColorRenderTargetLayoutsForPass(mDeferredPass),
      {mSceneLayout.get(), mCameraLayout.get(), mDeferredLayout.get()},
      numDirectionalLights, numPointLights);
  for (uint32_t i = 0; i < mCompositePasses.size(); ++i) {
    mCompositePasses[i]->createGraphicsPipeline(
        device, mRenderConfig->colorFormat, mRenderConfig->depthFormat,
        getColorRenderTargetLayoutsForPass(mDeferredPass),
        {mCompositeLayouts[i].get()});
  }
}

// std::vector<vk::PipelineLayout> ShaderManager::getPipelinesLayouts() {
//   std::vector<vk::PipelineLayout> layouts(mNumPasses);
//   layouts[0] = mGbufferPass->getPipelineLayout();
//   layouts[1] = mDeferredPass->getPipelineLayout();
//   int i = 2;
//   for (auto pass : mCompositePasses) {
//     layouts[i] = mCompositePasses[i]->getPipelineLayout();
//     ++i;
//   }
//   return layouts;
// }

std::vector<std::shared_ptr<BaseParser>> ShaderManager::getAllPasses() const {
  std::vector<std::shared_ptr<BaseParser>> allPasses;
  allPasses.push_back(mGbufferPass);
  allPasses.push_back(mDeferredPass);
  allPasses.insert(allPasses.end(), mCompositePasses.begin(),
                   mCompositePasses.end());
  return allPasses;
}

std::vector<vk::DescriptorSetLayout>
ShaderManager::getCompositeDescriptorSetLayout() const {
  std::vector<vk::DescriptorSetLayout> result;
  for (auto &l : mCompositeLayouts) {
    result.push_back(l.get());
  }
  return result;
}

} // namespace shader
} // namespace svulkan2
