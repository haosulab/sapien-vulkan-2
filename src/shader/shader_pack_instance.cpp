#include "svulkan2/shader/shader_pack_instance.h"
#include "svulkan2/common/launch_policy.h"
#include "svulkan2/core/context.h"
#include "svulkan2/shader/deferred.h"
#include "svulkan2/shader/gbuffer.h"
#include "svulkan2/shader/primitive.h"
#include "svulkan2/shader/primitive_shadow.h"
#include "svulkan2/shader/shadow.h"

namespace svulkan2 {
namespace shader {

ShaderPackInstance::ShaderPackInstance(ShaderPackInstanceDesc const &desc)
    : mDesc(desc) {
  // compile GLSL immediately
  mContext = core::Context::Get();
  mShaderPack =
      mContext->getResourceManager()->CreateShaderPack(mDesc.config->shaderDir);
}

static vk::UniqueDescriptorSetLayout
createObjectDescriptorSetLayout(vk::Device device) {
  auto binding = vk::DescriptorSetLayoutBinding(
      0, vk::DescriptorType::eUniformBuffer, 1,
      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
  return device.createDescriptorSetLayoutUnique(
      vk::DescriptorSetLayoutCreateInfo({}, binding));
}

static vk::UniqueDescriptorSetLayout
createCameraDescriptorSetLayout(vk::Device device) {
  auto binding = vk::DescriptorSetLayoutBinding(
      0, vk::DescriptorType::eUniformBuffer, 1,
      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
  return device.createDescriptorSetLayoutUnique(
      vk::DescriptorSetLayoutCreateInfo({}, binding));
}

static vk::UniqueDescriptorSetLayout createSceneDescriptorSetLayout(
    vk::Device device, DescriptorSetDescription const &sceneSetDescription) {
  std::vector<vk::DescriptorSetLayoutBinding> bindings;
  std::vector<vk::DescriptorBindingFlags> bindingFlags;
  for (uint32_t bindingIndex = 0;
       bindingIndex < sceneSetDescription.bindings.size(); ++bindingIndex) {
    if (sceneSetDescription.bindings.at(bindingIndex).type ==
        vk::DescriptorType::eUniformBuffer) {
      bindings.push_back({bindingIndex, vk::DescriptorType::eUniformBuffer, 1,
                          vk::ShaderStageFlagBits::eVertex |
                              vk::ShaderStageFlagBits::eFragment});
      bindingFlags.push_back({});
    } else if (sceneSetDescription.bindings.at(bindingIndex).type ==
               vk::DescriptorType::eCombinedImageSampler) {
      if (sceneSetDescription.bindings.at(bindingIndex).dim == 0) {
        bindings.push_back({bindingIndex,
                            vk::DescriptorType::eCombinedImageSampler, 1,
                            vk::ShaderStageFlagBits::eVertex |
                                vk::ShaderStageFlagBits::eFragment});
        bindingFlags.push_back({});
      } else {
        uint32_t arraySize =
            sceneSetDescription.bindings.at(bindingIndex).arraySize;
        bindings.push_back(
            {bindingIndex, vk::DescriptorType::eCombinedImageSampler, arraySize,
             vk::ShaderStageFlagBits::eVertex |
                 vk::ShaderStageFlagBits::eFragment});
        bindingFlags.push_back(vk::DescriptorBindingFlagBits::ePartiallyBound);
      }
    } else {
      throw std::runtime_error("invalid scene descriptor set");
    }
  }
  vk::DescriptorSetLayoutBindingFlagsCreateInfo flagInfo(bindingFlags.size(),
                                                         bindingFlags.data());
  vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings.size(),
                                               bindings.data());
  layoutInfo.setPNext(&flagInfo);
  return device.createDescriptorSetLayoutUnique(layoutInfo);
}

static vk::UniqueDescriptorSetLayout
createLightDescriptorSetLayout(vk::Device device) {
  auto binding =
      vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                     vk::ShaderStageFlagBits::eVertex);
  return device.createDescriptorSetLayoutUnique(
      vk::DescriptorSetLayoutCreateInfo({}, binding));
}

static std::vector<vk::UniqueDescriptorSetLayout>
createTextureDescriptorSetLayouts(
    vk::Device device, std::vector<std::shared_ptr<BaseParser>> const &passes) {
  std::vector<vk::UniqueDescriptorSetLayout> textureLayouts;
  for (auto &p : passes) {
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
    textureLayouts.push_back(std::move(layout));
  }
  return textureLayouts;
}

using optable_t =
    std::unordered_map<std::string,
                       std::vector<ShaderPack::RenderTargetOperation>>;

static ShaderPack::RenderTargetOperation
getPrevOperation(optable_t optable, std::string texName,
                 std::shared_ptr<BaseParser> pass) {
  for (int i = pass->getIndex() - 1; i >= 0; i--) {
    auto op = optable.at(texName)[i];
    if (op != ShaderPack::RenderTargetOperation::eNoOp) {
      return op;
    }
  }
  return ShaderPack::RenderTargetOperation::eNoOp;
}

static ShaderPack::RenderTargetOperation
getNextOperation(optable_t optable, std::string texName,
                 std::shared_ptr<BaseParser> pass) {
  auto ops = optable.at(texName);
  for (uint32_t i = pass->getIndex() + 1; i < ops.size(); i++) {
    auto op = ops.at(i);
    if (op != ShaderPack::RenderTargetOperation::eNoOp) {
      return op;
    }
  }
  return ShaderPack::RenderTargetOperation::eNoOp;
}

static ShaderPack::RenderTargetOperation getLastOperation(optable_t optable,
                                                          std::string texName) {
  auto ops = optable.at(texName);
  for (int i = ops.size() - 1; i >= 0; i--) {
    auto op = ops.at(i);
    if (op != ShaderPack::RenderTargetOperation::eNoOp) {
      return op;
    }
  }
  throw std::runtime_error("invalid last operation on " + texName);
}

static std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>>
getColorAttachmentLayoutsForPass(optable_t optable,
                                 std::shared_ptr<BaseParser> pass) {
  auto elems = pass->getTextureOutputLayout()->getElementsSorted();
  std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> result(elems.size());
  for (uint32_t i = 0; i < elems.size(); ++i) {
    ASSERT(elems[i].location == i,
           "output textures must have consecutive binding locations");
    auto texName = getOutTextureName(elems[i].name);
    auto prevOp = getPrevOperation(optable, texName, pass);
    auto nextOp = getNextOperation(optable, texName, pass);
    if (prevOp == ShaderPack::RenderTargetOperation::eNoOp) {
      result[i].first = vk::ImageLayout::eUndefined;
    } else if (prevOp == ShaderPack::RenderTargetOperation::eRead) {
      result[i].first = vk::ImageLayout::eShaderReadOnlyOptimal;
    } else {
      result[i].first = vk::ImageLayout::eColorAttachmentOptimal;
    }
    switch (nextOp) {
    case ShaderPack::RenderTargetOperation::eNoOp:
      result[i].second = vk::ImageLayout::eColorAttachmentOptimal;
      break;
    case ShaderPack::RenderTargetOperation::eRead:
      result[i].second = vk::ImageLayout::eShaderReadOnlyOptimal;
      break;
    case ShaderPack::RenderTargetOperation::eColorWrite:
      result[i].second = vk::ImageLayout::eColorAttachmentOptimal;
      break;
    case ShaderPack::RenderTargetOperation::eDepthWrite:
      throw std::runtime_error("invalid depth attachment layout");
    }
  }
  return result;
}

static std::pair<vk::ImageLayout, vk::ImageLayout>
getDepthAttachmentLayoutsForPass(optable_t optable,
                                 std::shared_ptr<BaseParser> pass) {
  auto name = pass->getDepthRenderTargetName();
  if (!name.has_value()) {
    return {vk::ImageLayout::eUndefined, vk::ImageLayout::eUndefined};
  }
  std::string texName = name.value();
  auto prevOp = getPrevOperation(optable, texName, pass);
  auto nextOp = getNextOperation(optable, texName, pass);
  vk::ImageLayout prev = vk::ImageLayout::eUndefined;
  vk::ImageLayout next = vk::ImageLayout::eUndefined;
  switch (prevOp) {
  case ShaderPack::RenderTargetOperation::eNoOp:
    prev = vk::ImageLayout::eUndefined;
    break;
  case ShaderPack::RenderTargetOperation::eRead:
  case ShaderPack::RenderTargetOperation::eDepthWrite:
    prev = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    break;
  case ShaderPack::RenderTargetOperation::eColorWrite:
    throw std::runtime_error("invalid depth attachment layout");
  }
  switch (nextOp) {
  case ShaderPack::RenderTargetOperation::eNoOp:
    next = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    break;
  case ShaderPack::RenderTargetOperation::eRead:
    next = vk::ImageLayout::eShaderReadOnlyOptimal;
    break;
  case ShaderPack::RenderTargetOperation::eDepthWrite:
    next = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    break;
  case ShaderPack::RenderTargetOperation::eColorWrite:
    throw std::runtime_error("invalid depth attachment layout");
  }
  return {prev, next};
}

static vk::Format
getRenderTargetFormat(RendererConfig const &config,
                      OutputDataLayout::Element const &element) {
  auto texname = getOutTextureName(element.name);
  if (config.textureFormat.contains(texname)) {
    return config.textureFormat.at(texname);
  }
  switch (element.dtype) {
  case DataType::eFLOAT:
    return config.colorFormat1;
    break;
  case DataType::eFLOAT4:
    return config.colorFormat4;
    break;
  case DataType::eUINT4:
    return (vk::Format::eR32G32B32A32Uint);
    break;
  default:
    throw std::runtime_error(
        "only float, float4 and uint4 are allowed in output attachments");
  }
}

std::future<void> ShaderPackInstance::loadAsync() {
  if (mLoaded) {
    return std::async(std::launch::deferred, []() {});
  }
  return std::async(LAUNCH_ASYNC, [this]() {
    std::lock_guard<std::mutex> lock(mLoadingMutex);
    if (mLoaded) {
      return;
    }

    vk::Device device = mContext->getDevice();

    auto textureOperationTable = mShaderPack->getTextureOperationTable();
    auto inputLayouts = mShaderPack->getShaderInputLayouts();
    auto passes = mShaderPack->getNonShadowPasses();

    // descriptor set layout
    mSceneSetLayout = createSceneDescriptorSetLayout(
        device, inputLayouts->sceneSetDescription);
    mObjectSetLayout = createObjectDescriptorSetLayout(device);
    mCameraSetLayout = createCameraDescriptorSetLayout(device);
    mLightSetLayout = createLightDescriptorSetLayout(device);
    mTextureSetLayouts = createTextureDescriptorSetLayouts(device, passes);

    // enable msaa only in the forward pipeline
    vk::SampleCountFlagBits msaa = vk::SampleCountFlagBits::e1;
    if (!mShaderPack->hasDeferredPass()) {
      msaa = mDesc.config->msaa;
    }

    // non shadow
    for (uint32_t passIdx = 0; passIdx < passes.size(); ++passIdx) {
      auto pass = passes[passIdx];
      std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
      for (auto type : pass->getUniformBindingTypes()) {
        switch (type) {
        case UniformBindingType::eCamera:
          descriptorSetLayouts.push_back(mCameraSetLayout.get());
          break;
        case UniformBindingType::eObject:
          descriptorSetLayouts.push_back(mObjectSetLayout.get());
          break;
        case UniformBindingType::eScene:
          descriptorSetLayouts.push_back(mSceneSetLayout.get());
          break;
        case UniformBindingType::eMaterial:
          descriptorSetLayouts.push_back(
              mContext->getMetallicDescriptorSetLayout());
          break;
        case UniformBindingType::eTextures:
          descriptorSetLayouts.push_back(mTextureSetLayouts[passIdx].get());
          break;
        default:
          throw std::runtime_error("ShaderManager::createPipelines: not "
                                   "implemented uniform binding type");
        }
      }
      std::vector<vk::Format> formats;
      for (auto layout : pass->getTextureOutputLayout()->getElementsSorted()) {
        formats.push_back(getRenderTargetFormat(*mDesc.config, layout));
      }

      // determine whether to use alpha blend
      bool alpha = false;
      if (auto p = std::dynamic_pointer_cast<GbufferPassParser>(pass)) {
        if (!mShaderPack->hasDeferredPass() ||
            p != mShaderPack->getGbufferPasses().at(0)) {
          alpha = true;
        }
      }

      PipelineResources res;
      res.layout = pass->createPipelineLayout(device, descriptorSetLayouts);
      res.renderPass = pass->createRenderPass(
          device, formats, mDesc.config->depthFormat,
          getColorAttachmentLayoutsForPass(textureOperationTable, pass),
          getDepthAttachmentLayoutsForPass(textureOperationTable, pass), msaa);
      res.pipeline = pass->createPipeline(
          device, res.layout.get(), res.renderPass.get(), mDesc.config->culling,
          vk::FrontFace::eCounterClockwise, alpha, msaa,
          mDesc.specializationConstants);

      mNonShadowPassResources.push_back(std::move(res));
    }

    // shadow
    auto shadowPass = mShaderPack->getShadowPass();
    if (shadowPass) {
      std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
      for (auto type : shadowPass->getUniformBindingTypes()) {
        switch (type) {
        case UniformBindingType::eObject:
          descriptorSetLayouts.push_back(mObjectSetLayout.get());
          break;
        case UniformBindingType::eLight:
          descriptorSetLayouts.push_back(mLightSetLayout.get());
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

      PipelineResources res;
      res.layout =
          shadowPass->createPipelineLayout(device, descriptorSetLayouts);
      res.renderPass = shadowPass->createRenderPass(
          device, {}, mDesc.config->depthFormat, {},
          {vk::ImageLayout::eUndefined,
           vk::ImageLayout::eShaderReadOnlyOptimal},
          vk::SampleCountFlagBits::e1);
      res.pipeline = shadowPass->createPipeline(
          device, res.layout.get(), res.renderPass.get(),
          vk::CullModeFlagBits::eFront, vk::FrontFace::eCounterClockwise, false,
          msaa, mDesc.specializationConstants);

      mShadowPassResources = std::move(res);
    }

    // point shadow
    auto pointShadowPass = mShaderPack->getPointShadowPass();
    if (pointShadowPass) {
      std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
      for (auto type : pointShadowPass->getUniformBindingTypes()) {
        switch (type) {
        case UniformBindingType::eObject:
          descriptorSetLayouts.push_back(mObjectSetLayout.get());
          break;
        case UniformBindingType::eLight:
          descriptorSetLayouts.push_back(mLightSetLayout.get());
          break;
        case UniformBindingType::eMaterial:
        case UniformBindingType::eCamera:
        case UniformBindingType::eScene:
          throw std::runtime_error(
              "pointShadow pass may only use object and light buffers");
        default:
          throw std::runtime_error("ShaderManager::createPipelines: not "
                                   "implemented uniform binding type");
        }
      }

      std::pair<vk::ImageLayout, vk::ImageLayout> layouts;
      if (shadowPass) {
        layouts = {vk::ImageLayout::eShaderReadOnlyOptimal,
                   vk::ImageLayout::eShaderReadOnlyOptimal};
      } else {
        layouts = {vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eShaderReadOnlyOptimal};
      }

      PipelineResources res;
      res.layout =
          pointShadowPass->createPipelineLayout(device, descriptorSetLayouts);
      res.renderPass = pointShadowPass->createRenderPass(
          device, {}, mDesc.config->depthFormat, {}, layouts,
          vk::SampleCountFlagBits::e1);
      res.pipeline = pointShadowPass->createPipeline(
          device, res.layout.get(), res.renderPass.get(),
          vk::CullModeFlagBits::eFront, vk::FrontFace::eCounterClockwise, false,
          msaa, mDesc.specializationConstants);

      mPointShadowPassResources = std::move(res);
    }

    // generate render target formats
    std::unordered_map<std::string, vk::Format> formats;
    for (auto pass : passes) {
      auto depthName = pass->getDepthRenderTargetName();
      if (depthName.has_value()) {
        mRenderTargetFormats[depthName.value()] = mDesc.config->depthFormat;
      }
      for (auto &elem : pass->getTextureOutputLayout()->elements) {
        std::string texName = getOutTextureName(elem.second.name);
        if (texName.ends_with("Depth")) {
          throw std::runtime_error(
              "You are not allowed to name your texture \"*Depth\"");
        }
        vk::Format texFormat =
            getRenderTargetFormat(*mDesc.config, elem.second);

        if (mRenderTargetFormats.find(texName) != mRenderTargetFormats.end()) {
          if (mRenderTargetFormats.at(texName) != texFormat) {
            std::runtime_error("Inconsistent texture format for \"" + texName +
                               "\"");
          }
        }
        mRenderTargetFormats[texName] = texFormat;
      }
    }

    // generate render target final layouts
    for (auto tex : mRenderTargetFormats) {
      auto op = getLastOperation(textureOperationTable, tex.first);
      switch (op) {
      case ShaderPack::RenderTargetOperation::eRead:
        mRenderTargetFinalLayouts[tex.first] =
            vk::ImageLayout::eShaderReadOnlyOptimal;
        break;
      case ShaderPack::RenderTargetOperation::eColorWrite:
        mRenderTargetFinalLayouts[tex.first] =
            vk::ImageLayout::eColorAttachmentOptimal;
        break;
      case ShaderPack::RenderTargetOperation::eDepthWrite:
        mRenderTargetFinalLayouts[tex.first] =
            vk::ImageLayout::eDepthStencilAttachmentOptimal;
        break;
      case ShaderPack::RenderTargetOperation::eNoOp:
        throw std::runtime_error("invalid render target");
      }
    }

    mLoaded = true;
  });
}

} // namespace shader
} // namespace svulkan2
