#include "svulkan2/shader/shader_manager.h"
#include "svulkan2/common/err.h"
#include <filesystem>
#include <set>

// #include <iostream>

namespace fs = std::filesystem;

namespace svulkan2 {
namespace shader {

ShaderManager::ShaderManager(std::shared_ptr<RendererConfig> config)
    : mRenderConfig(config) {
  mShaderConfig = std::make_shared<ShaderConfig>();
  processShadersInFolder(config->shaderDir);
}

void ShaderManager::processShadersInFolder(std::string const &folder) {
  fs::path path(folder);
  mNumPasses = 0;
  if (!fs::is_directory(path)) {
    throw std::runtime_error("[shader manager] " + folder +
                             " is not a directory");
  }
  if (!fs::is_regular_file(path / "gbuffer.vert")) {
    throw std::runtime_error("[shader manager] gbuffer.vert is required");
  }
  if (!fs::is_regular_file(path / "gbuffer.frag")) {
    throw std::runtime_error("[shader manager] gbuffer.frag is required");
  }
  if (!fs::is_regular_file(path / "deferred.vert")) {
    throw std::runtime_error("[shader manager] deferred.vert is required");
  }
  if (!fs::is_regular_file(path / "deferred.frag")) {
    throw std::runtime_error("[shader manager] deferred.frag is required");
  }

  GLSLCompiler::InitializeProcess();

  std::vector<std::future<void>> futures;

  // load gbuffer pass
  mGbufferPass = std::make_shared<GbufferPassParser>();
  std::string vsFile = (path / "gbuffer.vert").string();
  std::string fsFile = (path / "gbuffer.frag").string();
  futures.push_back(mGbufferPass->loadGLSLFilesAsync(vsFile, fsFile));
  mPassIndex[mGbufferPass] = mNumPasses++;

  // load deferred pass
  mDeferredPass = std::make_shared<DeferredPassParser>();
  vsFile = (path / "deferred.vert").string();
  fsFile = (path / "deferred.frag").string();
  futures.push_back(mDeferredPass->loadGLSLFilesAsync(vsFile, fsFile));
  mPassIndex[mDeferredPass] = mNumPasses++;

  int numCompositePasses = 0;
  for (const auto &entry : fs::directory_iterator(path)) {
    std::string filename = entry.path().filename().string();
    if (filename.substr(0, 9) == "composite" &&
        filename.substr(filename.length() - 5, 5) == ".frag")
      numCompositePasses++;
  }

  mCompositePasses.resize(numCompositePasses);
  vsFile = (path / "composite.vert").string();
  for (int i = 0; i < numCompositePasses; i++) {
    fsFile = path / ("composite" + std::to_string(i) + ".frag");
    mCompositePasses[i] = std::make_shared<CompositePassParser>();
    futures.push_back(mCompositePasses[i]->loadGLSLFilesAsync(vsFile, fsFile));
    mPassIndex[mCompositePasses[i]] = mNumPasses++;
  }
  for (auto &f : futures) {
    f.get();
  }

  GLSLCompiler::FinalizeProcess();

  populateShaderConfig();
  prepareRenderTargetFormats();
  prepareRenderTargetOperationTable();
}

void ShaderManager::populateShaderConfig() {
  auto allPasses = getAllPasses();
  for (auto &pass : allPasses) {
    pass->getUniformBindingTypes();
  }

  auto descs = mGbufferPass->getDescriptorSetDescriptions();
  for (auto &desc : descs) {
    switch (desc.type) {
    case UniformBindingType::eCamera:
      mShaderConfig->cameraBufferLayout =
          desc.buffers[desc.bindings[0].arrayIndex];
      break;
    case UniformBindingType::eObject:
      mShaderConfig->objectBufferLayout =
          desc.buffers[desc.bindings[0].arrayIndex];
      break;
    case UniformBindingType::eMaterial:
      if (desc.bindings[1].name == "colorTexture") {
        mShaderConfig->materialPipeline = ShaderConfig::eMETALLIC;
      } else {
        mShaderConfig->materialPipeline = ShaderConfig::eSPECULAR;
      }
      break;
    default:
      throw std::runtime_error("not implemented");
    }
  }

  mShaderConfig->vertexLayout = mGbufferPass->getVertexInputLayout();
  mShaderConfig->sceneBufferLayout = mDeferredPass->getSceneBufferLayout();
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
    mTextureOperationTable[tex.first] = std::vector<RenderTargetOperation>(
        mNumPasses, RenderTargetOperation::eNoOp);
  }

  // process gbuffer out textures:
  for (auto &elem : mGbufferPass->getTextureOutputLayout()->elements) {
    std::string texName = getOutTextureName(elem.second.name);
    mTextureOperationTable[texName][mPassIndex[mGbufferPass]] =
        RenderTargetOperation::eColorWrite;
  }
  mTextureOperationTable["Depth"][mPassIndex[mGbufferPass]] =
      RenderTargetOperation::eDepthWrite;

  // process input textures of deferred pass:
  for (auto &elem : mDeferredPass->getCombinedSamplerLayout()->elements) {
    std::string texName = getInTextureName(elem.second.name);
    if (mTextureOperationTable.find(texName) != mTextureOperationTable.end()) {
      mTextureOperationTable[texName][mPassIndex[mDeferredPass]] =
          RenderTargetOperation::eRead;
    }
  }

  // process out textures of deferred paas:
  for (auto &elem : mDeferredPass->getTextureOutputLayout()->elements) {
    std::string texName = getOutTextureName(elem.second.name);
    mTextureOperationTable[texName][mPassIndex[mDeferredPass]] =
        RenderTargetOperation::eColorWrite;
  }

  for (uint32_t i = 0; i < mCompositePasses.size(); i++) {
    auto compositePass = mCompositePasses[i];
    // process input textures of composite pass:
    for (auto &elem : compositePass->getCombinedSamplerLayout()->elements) {
      std::string texName = getInTextureName(elem.second.name);
      if (mTextureOperationTable.find(texName) !=
          mTextureOperationTable.end()) {
        mTextureOperationTable[texName][mPassIndex[mCompositePasses[i]]] =
            RenderTargetOperation::eRead;
      }
    }
    // add composite out texture to the set:
    for (auto &elem : compositePass->getTextureOutputLayout()->elements) {
      std::string texName = getOutTextureName(elem.second.name);
      mTextureOperationTable[texName][mPassIndex[mCompositePasses[i]]] =
          RenderTargetOperation::eColorWrite;
    }
  }
  // for (auto &[tex, ops] : mTextureOperationTable) {
  //   std::cout << std::setw(20) << tex << " ";
  //   for (auto &op : ops) {
  //     if (op == RenderTargetOperation::eNoOp) {
  //       std::cout << "N ";
  //     } else if (op == RenderTargetOperation::eRead) {
  //       std::cout << "R ";
  //     } else if (op == RenderTargetOperation::eWrite) {
  //       std::cout << "W ";
  //     }
  //   }
  //   std::cout << std::endl;
  // }
}

RenderTargetOperation
ShaderManager::getNextOperation(std::string texName,
                                std::shared_ptr<BaseParser> pass) {
  for (uint32_t i = mPassIndex[pass] + 1; i < mNumPasses; i++) {
    RenderTargetOperation op = mTextureOperationTable[texName][i];
    if (op != RenderTargetOperation::eNoOp) {
      return op;
    }
  }
  return RenderTargetOperation::eNoOp;
}

RenderTargetOperation
ShaderManager::getPrevOperation(std::string texName,
                                std::shared_ptr<BaseParser> pass) {
  for (int i = mPassIndex[pass] - 1; i >= 0; i--) {
    RenderTargetOperation op = mTextureOperationTable[texName][i];
    if (op != RenderTargetOperation::eNoOp) {
      return op;
    }
  }
  return RenderTargetOperation::eNoOp;
}

RenderTargetOperation
ShaderManager::getLastOperation(std::string texName) const {
  for (int i = mNumPasses - 1; i >= 0; i--) {
    RenderTargetOperation op = mTextureOperationTable.at(texName).at(i);
    if (op != RenderTargetOperation::eNoOp) {
      return op;
    }
  }
  throw std::runtime_error("invalid last operation on " + texName);
}

std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>>
ShaderManager::getColorAttachmentLayoutsForPass(
    std::shared_ptr<BaseParser> pass) {
  auto elems = pass->getTextureOutputLayout()->getElementsSorted();
  std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> result(elems.size());
  for (uint32_t i = 0; i < elems.size(); ++i) {
    ASSERT(elems[i].location == i,
           "output textures must have consecutive binding locations");
    auto texName = getOutTextureName(elems[i].name);
    auto prevOp = getPrevOperation(texName, pass);
    auto nextOp = getNextOperation(texName, pass);
    if (prevOp == RenderTargetOperation::eNoOp) {
      result[i].first = vk::ImageLayout::eUndefined;
    } else {
      result[i].first = vk::ImageLayout::eColorAttachmentOptimal;
    }
    switch (nextOp) {
    case RenderTargetOperation::eNoOp:
      result[i].second = vk::ImageLayout::eColorAttachmentOptimal;
      break;
    case RenderTargetOperation::eRead:
      result[i].second = vk::ImageLayout::eShaderReadOnlyOptimal;
      break;
    case RenderTargetOperation::eColorWrite:
      result[i].second = vk::ImageLayout::eColorAttachmentOptimal;
      break;
    case RenderTargetOperation::eDepthWrite:
      throw std::runtime_error("invalid depth attachment layout");
    }
  }
  return result;
}

std::pair<vk::ImageLayout, vk::ImageLayout>
ShaderManager::getDepthAttachmentLayoutsForPass(
    std::shared_ptr<BaseParser> pass) {
  std::string texName;
  if (pass == mGbufferPass) {
    texName = "Depth";
  }
  auto prevOp = getPrevOperation(texName, pass);
  auto nextOp = getNextOperation(texName, pass);
  vk::ImageLayout prev;
  vk::ImageLayout next;
  switch (prevOp) {
  case RenderTargetOperation::eNoOp:
    prev = vk::ImageLayout::eUndefined;
    break;
  case RenderTargetOperation::eRead:
  case RenderTargetOperation::eDepthWrite:
    prev = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    break;
  case RenderTargetOperation::eColorWrite:
    throw std::runtime_error("invalid depth attachment layout");
  }
  switch (nextOp) {
  case RenderTargetOperation::eNoOp:
    next = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    break;
  case RenderTargetOperation::eRead:
    next = vk::ImageLayout::eShaderReadOnlyOptimal;
    break;
  case RenderTargetOperation::eDepthWrite:
    next = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    break;
  case RenderTargetOperation::eColorWrite:
    throw std::runtime_error("invalid depth attachment layout");
  }
  return {prev, next};
}

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
      getColorAttachmentLayoutsForPass(mGbufferPass),
      getDepthAttachmentLayoutsForPass(mGbufferPass),
      {mCameraLayout.get(), mObjectLayout.get(),
       mShaderConfig->materialPipeline ==
               ShaderConfig::MaterialPipeline::eMETALLIC
           ? context.getMetallicDescriptorSetLayout()
           : context.getSpecularDescriptorSetLayout()});

  mDeferredPass->createGraphicsPipeline(
      device, mRenderConfig->colorFormat,
      getColorAttachmentLayoutsForPass(mDeferredPass),
      {mSceneLayout.get(), mCameraLayout.get(), mDeferredLayout.get()},
      numDirectionalLights, numPointLights);
  for (uint32_t i = 0; i < mCompositePasses.size(); ++i) {
    mCompositePasses[i]->createGraphicsPipeline(
        device, mRenderConfig->colorFormat,
        getColorAttachmentLayoutsForPass(mCompositePasses[i]),
        {mCompositeLayouts[i].get()});
  }
}

std::vector<vk::DescriptorSetLayout>
ShaderManager::getCompositeDescriptorSetLayouts() const {
  std::vector<vk::DescriptorSetLayout> result;
  for (auto &l : mCompositeLayouts) {
    result.push_back(l.get());
  }
  return result;
}

std::vector<std::shared_ptr<BaseParser>> ShaderManager::getAllPasses() const {
  std::vector<std::shared_ptr<BaseParser>> allPasses;
  allPasses.push_back(mGbufferPass);
  allPasses.push_back(mDeferredPass);
  allPasses.insert(allPasses.end(), mCompositePasses.begin(),
                   mCompositePasses.end());
  return allPasses;
}

std::unordered_map<std::string, vk::ImageLayout>
ShaderManager::getRenderTargetFinalLayouts() const {
  std::unordered_map<std::string, vk::ImageLayout> result;
  for (auto tex : mRenderTargetFormats) {
    auto op = getLastOperation(tex.first);
    switch (op) {
    case RenderTargetOperation::eRead:
      result[tex.first] = vk::ImageLayout::eShaderReadOnlyOptimal;
      break;
    case RenderTargetOperation::eColorWrite:
      result[tex.first] = vk::ImageLayout::eColorAttachmentOptimal;
      break;
    case RenderTargetOperation::eDepthWrite:
      result[tex.first] = vk::ImageLayout::eDepthStencilAttachmentOptimal;
      break;
    case RenderTargetOperation::eNoOp:
      throw std::runtime_error("invalid render target");
    }
  }
  return result;
}

} // namespace shader
} // namespace svulkan2
