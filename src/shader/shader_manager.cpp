#include "svulkan2/shader/shader_manager.h"
#include "svulkan2/common/err.h"
#include <filesystem>
#include <iostream>
#include <set>

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

  GLSLCompiler::InitializeProcess();

  std::vector<std::future<void>> futures;

  if (fs::is_regular_file(path / "shadow.vert")) {
    mShadowEnabled = true;
    mShadowPass = std::make_shared<ShadowPassParser>();
    mShadowPass->setName("Shadow");
    futures.push_back(
        mShadowPass->loadGLSLFilesAsync((path / "shadow.vert").string(), ""));
  }

  bool hasDeferred = fs::is_regular_file(path / "deferred.vert") &&
                     fs::is_regular_file(path / "deferred.frag");
  mAllPasses = {};

  mNumGbufferPasses = 0;
  for (const auto &entry : fs::directory_iterator(path)) {
    std::string filename = entry.path().filename().string();
    if (filename.starts_with("gbuffer") &&
        filename.ends_with(".frag")) {
      mNumGbufferPasses++;
    }
  }

  std::shared_ptr<GbufferPassParser> firstGbufferPass;
  for (uint32_t i = 0; i < mNumGbufferPasses; ++i) {
    std::string suffix = i == 0 ? "" : std::to_string(i);
    // load gbuffer pass
    auto gbufferPass = std::make_shared<GbufferPassParser>();
    if (i == 0) {
      firstGbufferPass = gbufferPass;
    }
    if (!hasDeferred || i != 0) {
      gbufferPass->enableAlphaBlend(true);
    }
    gbufferPass->setName("Gbuffer" + suffix);
    mAllPasses.push_back(gbufferPass);
    mPassIndex[gbufferPass] = mAllPasses.size() - 1;

    std::string vsFile = (path / ("gbuffer" + suffix + ".vert")).string();
    std::string fsFile = (path / ("gbuffer" + suffix + ".frag")).string();
    futures.push_back(gbufferPass->loadGLSLFilesAsync(vsFile, fsFile));
  }

  {
    std::string vsFile = (path / "ao.vert").string();
    std::string fsFile = (path / "ao.frag").string();
    if (fs::is_regular_file(vsFile) && fs::is_regular_file(fsFile)) {
      auto aoPass = std::make_shared<DeferredPassParser>();
      aoPass->setName("AO");
      mAllPasses.push_back(aoPass);
      mPassIndex[aoPass] = mAllPasses.size() - 1;
      futures.push_back(aoPass->loadGLSLFilesAsync(vsFile, fsFile));
    }
  }

  // load deferred pass
  std::string vsFile = (path / "deferred.vert").string();
  std::string fsFile = (path / "deferred.frag").string();
  if (fs::is_regular_file(vsFile) && fs::is_regular_file(fsFile)) {
    auto deferredPass = std::make_shared<DeferredPassParser>();
    deferredPass->setName("Deferred");
    mAllPasses.push_back(deferredPass);
    mPassIndex[deferredPass] = mAllPasses.size() - 1;
    futures.push_back(deferredPass->loadGLSLFilesAsync(vsFile, fsFile));
  }

  int numCompositePasses = 0;
  for (const auto &entry : fs::directory_iterator(path)) {
    std::string filename = entry.path().filename().string();
    if (filename.starts_with("composite") && filename.ends_with(".frag"))
      numCompositePasses++;
  }

  vsFile = (path / "composite.vert").string();
  for (int i = 0; i < numCompositePasses; i++) {
    fsFile = path / ("composite" + std::to_string(i) + ".frag");
    auto compositePass = std::make_shared<DeferredPassParser>();
    compositePass->setName("Composite" + std::to_string(i));
    mAllPasses.push_back(compositePass);
    mPassIndex[compositePass] = mAllPasses.size() - 1;
    futures.push_back(compositePass->loadGLSLFilesAsync(vsFile, fsFile));
  }
  for (auto &f : futures) {
    f.get();
  }

  GLSLCompiler::FinalizeProcess();

  mShaderConfig->vertexLayout = firstGbufferPass->getVertexInputLayout();

  populateShaderConfig();
  prepareRenderTargetFormats();
  prepareRenderTargetOperationTable();
}

void ShaderManager::populateShaderConfig() {
  auto allPasses = getAllPasses();
  ShaderConfig::MaterialPipeline material =
      ShaderConfig::MaterialPipeline::eUNKNOWN;
  if (mShadowEnabled) {
    allPasses.push_back(mShadowPass);
  }
  for (auto &pass : allPasses) {
    auto descs = pass->getDescriptorSetDescriptions();
    for (auto &desc : descs) {
      switch (desc.type) {
      case UniformBindingType::eCamera:
        if (mCameraSetDesc.type == UniformBindingType::eUnknown) {
          mCameraSetDesc = desc;
        } else {
          mCameraSetDesc = mCameraSetDesc.merge(desc);
        }
        break;
      case UniformBindingType::eObject:
        if (mObjectSetDesc.type == UniformBindingType::eUnknown) {
          mObjectSetDesc = desc;
        } else {
          mObjectSetDesc = mObjectSetDesc.merge(desc);
        }
        break;
      case UniformBindingType::eScene:
        if (mSceneSetDesc.type == UniformBindingType::eUnknown) {
          mSceneSetDesc = desc;
        } else {
          mSceneSetDesc = mSceneSetDesc.merge(desc);
        }
        break;
      case UniformBindingType::eLight:
        if (mLightSetDesc.type == UniformBindingType::eUnknown) {
          mLightSetDesc = desc;
        } else {
          mLightSetDesc = mLightSetDesc.merge(desc);
        }
        break;
      case UniformBindingType::eMaterial:
        if (desc.bindings[1].name == "colorTexture") {
          if (material == ShaderConfig::MaterialPipeline::eSPECULAR) {
            throw std::runtime_error(
                "Only one type of material (either metallic or specular) may "
                "be used in all shader pipelines.");
          }
          material = ShaderConfig::MaterialPipeline::eMETALLIC;
        } else {
          if (material == ShaderConfig::MaterialPipeline::eMETALLIC) {
            throw std::runtime_error(
                "Only one type of material (either metallic or specular) may "
                "be used in all shader pipelines.");
          }
          material = ShaderConfig::MaterialPipeline::eMETALLIC;
        }
        break;
      case UniformBindingType::eTextures:
        break;
      case UniformBindingType::eNone:
      case UniformBindingType::eUnknown:
        throw std::runtime_error("invalid descriptor set");
      }
    }
  }

  if (material == ShaderConfig::MaterialPipeline::eUNKNOWN) {
    throw std::runtime_error(
        "at least one shader needs to specify the material buffer");
  }
  mShaderConfig->materialPipeline = material;
  mShaderConfig->cameraBufferLayout =
      mCameraSetDesc.buffers[mCameraSetDesc.bindings.at(0).arrayIndex];
  mShaderConfig->objectBufferLayout =
      mObjectSetDesc.buffers[mObjectSetDesc.bindings.at(0).arrayIndex];
  mShaderConfig->sceneBufferLayout =
      mSceneSetDesc.buffers[mSceneSetDesc.bindings.at(0).arrayIndex];
  if (mShadowEnabled) {
    mShaderConfig->lightBufferLayout =
        mLightSetDesc.buffers[mLightSetDesc.bindings.at(0).arrayIndex];

    for (auto &binding : mSceneSetDesc.bindings) {
      if (binding.second.name == "ShadowBuffer") {
        mShaderConfig->shadowBufferLayout =
            mSceneSetDesc.buffers[binding.second.arrayIndex];
        break;
      }
    }
    if (!mShaderConfig->shadowBufferLayout) {
      throw std::runtime_error("Scene must declare ShadowBuffer");
    }
  }
}

void ShaderManager::prepareRenderTargetFormats() {
  auto allPasses = getAllPasses();
  for (auto pass : allPasses) {
    auto depthName = pass->getDepthRenderTargetName();
    if (depthName.has_value()) {
      mRenderTargetFormats[depthName.value()] = mRenderConfig->depthFormat;
    }
    for (auto &elem : pass->getTextureOutputLayout()->elements) {
      std::string texName = getOutTextureName(elem.second.name);
      if (texName.ends_with("Depth")) {
        throw std::runtime_error(
            "You are not allowed to name your texture \"*Depth\"");
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
  auto passes = getAllPasses();
  for (auto tex : mRenderTargetFormats) {
    mTextureOperationTable[tex.first] = std::vector<RenderTargetOperation>(
        passes.size(), RenderTargetOperation::eNoOp);
  }

  for (uint32_t passIdx = 0; passIdx < passes.size(); ++passIdx) {
    auto pass = passes[passIdx];
    for (auto name : pass->getInputTextureNames()) {
      if (mTextureOperationTable.find(name) != mTextureOperationTable.end()) {
        mTextureOperationTable[name][passIdx] = RenderTargetOperation::eRead;
      }
    }
    for (auto &name : pass->getColorRenderTargetNames()) {
      mTextureOperationTable[name][passIdx] =
          RenderTargetOperation::eColorWrite;
    }
    auto name = pass->getDepthRenderTargetName();
    if (name.has_value()) {
      mTextureOperationTable[name.value()][passIdx] =
          RenderTargetOperation::eDepthWrite;
    }
  }

  std::stringstream ss;
  ss << "Operation Table" << std::endl;
  ss << std::setw(20) << ""
     << " ";
  for (auto pass : getAllPasses()) {
    ss << std::setw(15) << pass->getName() << " ";
  }
  ss << std::endl;
  for (auto &[tex, ops] : mTextureOperationTable) {
    ss << std::setw(20) << tex << " ";
    for (auto &op : ops) {
      if (op == RenderTargetOperation::eNoOp) {
        ss << std::setw(15) << "N"
           << " ";
      } else if (op == RenderTargetOperation::eRead) {
        ss << std::setw(15) << "R"
           << " ";
      } else if (op == RenderTargetOperation::eColorWrite) {
        ss << std::setw(15) << "W"
           << " ";
      } else if (op == RenderTargetOperation::eDepthWrite) {
        ss << std::setw(15) << "D"
           << " ";
      }
    }
    ss << std::endl;
  }
  log::info(ss.str());
}

RenderTargetOperation
ShaderManager::getNextOperation(std::string texName,
                                std::shared_ptr<BaseParser> pass) {
  for (uint32_t i = mPassIndex[pass] + 1; i < mAllPasses.size(); i++) {
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
  for (int i = mAllPasses.size() - 1; i >= 0; i--) {
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
    } else if (prevOp == RenderTargetOperation::eRead) {
      result[i].first = vk::ImageLayout::eShaderReadOnlyOptimal;
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
  auto name = pass->getDepthRenderTargetName();
  if (!name.has_value()) {
    return {vk::ImageLayout::eUndefined, vk::ImageLayout::eUndefined};
  }
  std::string texName = name.value();
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
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (uint32_t bindingIndex = 0;
         bindingIndex < mSceneSetDesc.bindings.size(); ++bindingIndex) {
      if (mSceneSetDesc.bindings.at(bindingIndex).type ==
          vk::DescriptorType::eUniformBuffer) {
        bindings.push_back({bindingIndex, vk::DescriptorType::eUniformBuffer, 1,
                            vk::ShaderStageFlagBits::eVertex |
                                vk::ShaderStageFlagBits::eFragment});
      } else if (mSceneSetDesc.bindings.at(bindingIndex).type ==
                 vk::DescriptorType::eCombinedImageSampler) {
        bindings.push_back({bindingIndex,
                            vk::DescriptorType::eCombinedImageSampler, 1,
                            vk::ShaderStageFlagBits::eVertex |
                                vk::ShaderStageFlagBits::eFragment});
      } else {
        throw std::runtime_error("invalid scene descriptor set");
      }
    }
    mSceneLayout = device.createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo({}, bindings.size(),
                                          bindings.data()));
  }
  {
    vk::DescriptorSetLayoutBinding binding(0,
                                           vk::DescriptorType::eUniformBuffer,
                                           1, vk::ShaderStageFlagBits::eVertex);
    mLightLayout = device.createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo({}, 1, &binding));
  }

  {
    mInputTextureLayouts.clear();
    auto parsers = getAllPasses();
    for (auto &p : parsers) {
      auto types = p->getUniformBindingTypes();
      vk::UniqueDescriptorSetLayout layout{};
      for (auto type : types) {
        if (type == shader::UniformBindingType::eTextures) {
          std::vector<vk::DescriptorSetLayoutBinding> bindings;
          auto texs = p->getInputTextureNames();
          for (uint32_t i = 0; i < texs.size(); ++i) {
            bindings.push_back(vk::DescriptorSetLayoutBinding(
                i, vk::DescriptorType::eCombinedImageSampler, 1,
                vk::ShaderStageFlagBits::eFragment));
          }
          layout = device.createDescriptorSetLayoutUnique(
              vk::DescriptorSetLayoutCreateInfo({}, bindings.size(),
                                                bindings.data()));
          break;
        }
      }
      mInputTextureLayouts.push_back(std::move(layout));
    }
  }
}

void ShaderManager::createPipelines(
    std::shared_ptr<core::Context> context,
    std::map<std::string, SpecializationConstantValue> const
        &specializationConstantInfo) {
  auto device = context->getDevice();

  if (not mDescriptorSetLayoutsCreated) {
    createDescriptorSetLayouts(device);
    mDescriptorSetLayoutsCreated = true;
  }

  auto passes = getAllPasses();
  for (uint32_t passIdx = 0; passIdx < passes.size(); ++passIdx) {
    auto pass = passes[passIdx];
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
    for (auto type : pass->getUniformBindingTypes()) {
      switch (type) {
      case UniformBindingType::eCamera:
        descriptorSetLayouts.push_back(mCameraLayout.get());
        break;
      case UniformBindingType::eObject:
        descriptorSetLayouts.push_back(mObjectLayout.get());
        break;
      case UniformBindingType::eScene:
        descriptorSetLayouts.push_back(mSceneLayout.get());
        break;
      case UniformBindingType::eMaterial:
        descriptorSetLayouts.push_back(
            mShaderConfig->materialPipeline ==
                    ShaderConfig::MaterialPipeline::eMETALLIC
                ? context->getMetallicDescriptorSetLayout()
                : context->getSpecularDescriptorSetLayout());
        break;
      case UniformBindingType::eTextures:
        descriptorSetLayouts.push_back(mInputTextureLayouts[passIdx].get());
        break;
      default:
        throw std::runtime_error("ShaderManager::createPipelines: not "
                                 "implemented uniform binding type");
      }
    }
    pass->createGraphicsPipeline(
        device, mRenderConfig->colorFormat, mRenderConfig->depthFormat,
        mRenderConfig->culling, vk::FrontFace::eCounterClockwise,
        getColorAttachmentLayoutsForPass(pass),
        getDepthAttachmentLayoutsForPass(pass), descriptorSetLayouts,
        specializationConstantInfo);
  }

  if (mShadowEnabled) {
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
    for (auto type : mShadowPass->getUniformBindingTypes()) {
      switch (type) {
      case UniformBindingType::eObject:
        descriptorSetLayouts.push_back(mObjectLayout.get());
        break;
      case UniformBindingType::eLight:
        descriptorSetLayouts.push_back(mLightLayout.get());
        break;
      case UniformBindingType::eMaterial:
      case UniformBindingType::eCamera:
      case UniformBindingType::eScene:
        throw std::runtime_error(
            "shadow pass may only use object and light buffers");

      default:
        throw std::runtime_error("ShaderManager::createPipelines: not "
                                 "implemented uniform binding type");
      }
    }
    mShadowPass->createGraphicsPipeline(
        device, mRenderConfig->colorFormat, mRenderConfig->depthFormat,
        mRenderConfig->culling, vk::FrontFace::eCounterClockwise, {},
        {vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        descriptorSetLayouts, specializationConstantInfo);
  }
}

std::vector<vk::DescriptorSetLayout>
ShaderManager::getInputTextureLayouts() const {
  std::vector<vk::DescriptorSetLayout> result;
  for (auto &l : mInputTextureLayouts) {
    result.push_back(l.get());
  }
  return result;
}

std::vector<std::shared_ptr<BaseParser>> ShaderManager::getAllPasses() const {
  return mAllPasses;
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
