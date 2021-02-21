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

  bool hasDeferred = fs::is_regular_file(path / "deferred.vert") &&
                     fs::is_regular_file(path / "deferred.frag");

  mAllPasses = {};
  // load gbuffer pass
  auto gbufferPass = std::make_shared<GbufferPassParser>();
  if (!hasDeferred) {
    gbufferPass->enableAlphaBlend(true);
  }
  mAllPasses.push_back(gbufferPass);
  mPassIndex[gbufferPass] = mAllPasses.size() - 1;

  std::string vsFile = (path / "gbuffer.vert").string();
  std::string fsFile = (path / "gbuffer.frag").string();
  futures.push_back(gbufferPass->loadGLSLFilesAsync(vsFile, fsFile));

  // load deferred pass
  vsFile = (path / "deferred.vert").string();
  fsFile = (path / "deferred.frag").string();
  if (fs::is_regular_file(vsFile) && fs::is_regular_file(fsFile)) {
    auto deferredPass = std::make_shared<DeferredPassParser>();
    mAllPasses.push_back(deferredPass);
    mPassIndex[deferredPass] = mAllPasses.size() - 1;
    futures.push_back(deferredPass->loadGLSLFilesAsync(vsFile, fsFile));
  }

  int numCompositePasses = 0;
  for (const auto &entry : fs::directory_iterator(path)) {
    std::string filename = entry.path().filename().string();
    if (filename.substr(0, 9) == "composite" &&
        filename.substr(filename.length() - 5, 5) == ".frag")
      numCompositePasses++;
  }

  vsFile = (path / "composite.vert").string();
  for (int i = 0; i < numCompositePasses; i++) {
    fsFile = path / ("composite" + std::to_string(i) + ".frag");
    auto compositePass = std::make_shared<DeferredPassParser>();
    mAllPasses.push_back(compositePass);
    mPassIndex[compositePass] = mAllPasses.size() - 1;
    futures.push_back(compositePass->loadGLSLFilesAsync(vsFile, fsFile));
  }
  for (auto &f : futures) {
    f.get();
  }

  GLSLCompiler::FinalizeProcess();

  mShaderConfig->vertexLayout = gbufferPass->getVertexInputLayout();

  populateShaderConfig();
  prepareRenderTargetFormats();
  prepareRenderTargetOperationTable();
}

void ShaderManager::populateShaderConfig() {
  auto allPasses = getAllPasses();
  DescriptorSetDescription cameraSetDesc;
  DescriptorSetDescription objectSetDesc;
  DescriptorSetDescription sceneSetDesc;
  DescriptorSetDescription lightSetDesc;
  ShaderConfig::MaterialPipeline material =
      ShaderConfig::MaterialPipeline::eUNKNOWN;

  for (auto &pass : allPasses) {
    auto descs = pass->getDescriptorSetDescriptions();
    for (auto &desc : descs) {
      switch (desc.type) {
      case UniformBindingType::eCamera:
        if (cameraSetDesc.type == UniformBindingType::eUnknown) {
          cameraSetDesc = desc;
        } else {
          cameraSetDesc = cameraSetDesc.merge(desc);
        }
        break;
      case UniformBindingType::eObject:
        if (objectSetDesc.type == UniformBindingType::eUnknown) {
          objectSetDesc = desc;
        } else {
          objectSetDesc = objectSetDesc.merge(desc);
        }
        break;
      case UniformBindingType::eScene:
        if (sceneSetDesc.type == UniformBindingType::eUnknown) {
          sceneSetDesc = desc;
        } else {
          sceneSetDesc = sceneSetDesc.merge(desc);
        }
        break;
      case UniformBindingType::eLight:
        if (lightSetDesc.type == UniformBindingType::eUnknown) {
          lightSetDesc = desc;
        } else {
          lightSetDesc = lightSetDesc.merge(desc);
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
      cameraSetDesc.buffers[cameraSetDesc.bindings[0].arrayIndex];
  mShaderConfig->objectBufferLayout =
      objectSetDesc.buffers[objectSetDesc.bindings[0].arrayIndex];
  mShaderConfig->sceneBufferLayout =
      sceneSetDesc.buffers[sceneSetDesc.bindings[0].arrayIndex];
  // TODO: light buffer
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

  for (auto &[tex, ops] : mTextureOperationTable) {
    std::cout << std::setw(20) << tex << " ";
    for (auto &op : ops) {
      if (op == RenderTargetOperation::eNoOp) {
        std::cout << "N ";
      } else if (op == RenderTargetOperation::eRead) {
        std::cout << "R ";
      } else if (op == RenderTargetOperation::eColorWrite) {
        std::cout << "W ";
      } else if (op == RenderTargetOperation::eDepthWrite) {
        std::cout << "D ";
      }
    }
    std::cout << std::endl;
  }
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
    vk::DescriptorSetLayoutBinding binding(
        0, vk::DescriptorType::eUniformBuffer, 1,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
    mSceneLayout = device.createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo({}, 1, &binding));
  }

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

void ShaderManager::createPipelines(
    core::Context &context,
    std::map<std::string, SpecializationConstantValue> const
        &specializationConstantInfo) {
  auto device = context.getDevice();
  createDescriptorSetLayouts(device);

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
                ? context.getMetallicDescriptorSetLayout()
                : context.getSpecularDescriptorSetLayout());
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
