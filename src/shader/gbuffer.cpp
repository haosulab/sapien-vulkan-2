#include "svulkan2/shader/gbuffer.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {
namespace shader {

void GbufferPassParser::reflectSPV() {
  spirv_cross::Compiler vertComp(mVertSPVCode);
  std::vector<DescriptorSetDescription> vertDesc;
  std::vector<DescriptorSetDescription> fragDesc;
  try {
    mVertexInputLayout = parseVertexInput(vertComp);
    for (uint32_t i = 0; i < 4; ++i) {
      vertDesc.push_back(getDescriptorSetDescription(vertComp, i));
    }
  } catch (std::runtime_error const &err) {
    throw std::runtime_error("[vert]" + std::string(err.what()));
  }

  spirv_cross::Compiler fragComp(mFragSPVCode);
  try {
    mSpecializationConstantLayout = parseSpecializationConstant(fragComp);
    mTextureOutputLayout = parseTextureOutput(fragComp);
    for (uint32_t i = 0; i < 4; ++i) {
      fragDesc.push_back(getDescriptorSetDescription(fragComp, i));
    }
  } catch (std::runtime_error const &err) {
    throw std::runtime_error("[frag]" + std::string(err.what()));
  }

  for (uint32_t i = 0, stop = false; i < 4; ++i) {
    if (vertDesc[i].type == UniformBindingType::eNone &&
        fragDesc[i].type == UniformBindingType::eNone) {
      stop = true;
      continue;
    }
    if (stop == true) {
      throw std::runtime_error("gbuffer: descriptor sets should use "
                               "consecutive integers starting from 0");
    }
    if (vertDesc[i].type == UniformBindingType::eNone) {
      mDescriptorSetDescriptions.push_back(fragDesc[i]);
    } else if (fragDesc[i].type == UniformBindingType::eNone) {
      mDescriptorSetDescriptions.push_back(vertDesc[i]);
    } else {
      mDescriptorSetDescriptions.push_back(vertDesc[i].merge(fragDesc[i]));
    }
  }

  validate();
}

void GbufferPassParser::validate() const {
  for (auto &elem : mTextureOutputLayout->elements) {
    ASSERT(elem.second.name.substr(0, 3) == "out",
           "[frag]all out texture variables must start with \"out\"");
  }
};

vk::PipelineLayout GbufferPassParser::createPipelineLayout(
    vk::Device device,
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts) {
  vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
  pipelineLayoutInfo.setLayoutCount = descriptorSetLayouts.size();
  pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
  mPipelineLayout = device.createPipelineLayoutUnique(pipelineLayoutInfo);
  return mPipelineLayout.get();
}

vk::RenderPass GbufferPassParser::createRenderPass(
    vk::Device device, vk::Format colorFormat, vk::Format depthFormat,
    std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
        &colorTargetLayouts,
    std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout) {
  std::vector<vk::AttachmentDescription> attachmentDescriptions;
  std::vector<vk::AttachmentReference> colorAttachments;

  auto elems = mTextureOutputLayout->getElementsSorted();
  for (uint32_t i = 0; i < elems.size(); ++i) {
    colorAttachments.push_back({i, vk::ImageLayout::eColorAttachmentOptimal});
    vk::Format format;
    if (elems[i].dtype == eFLOAT4) {
      format = colorFormat;
    } else if (elems[i].dtype == eUINT4) {
      format = vk::Format::eR32G32B32A32Uint;
    } else {
      throw std::runtime_error(
          "only float4 and uint4 are allowed in output attachments");
    }

    attachmentDescriptions.push_back(vk::AttachmentDescription(
        {}, format, vk::SampleCountFlagBits::e1,
        colorTargetLayouts[i].first == vk::ImageLayout::eUndefined
            ? vk::AttachmentLoadOp::eClear
            : vk::AttachmentLoadOp::eLoad,
        vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eDontCare, colorTargetLayouts[i].first,
        colorTargetLayouts[i].second));
  }
  attachmentDescriptions.push_back(vk::AttachmentDescription(
      vk::AttachmentDescriptionFlags(), depthFormat,
      vk::SampleCountFlagBits::e1,
      depthLayout.first == vk::ImageLayout::eUndefined
          ? vk::AttachmentLoadOp::eClear
          : vk::AttachmentLoadOp::eLoad,
      vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
      vk::AttachmentStoreOp::eDontCare, depthLayout.first, depthLayout.second));

  vk::AttachmentReference depthAttachment(
      elems.size(), vk::ImageLayout::eDepthStencilAttachmentOptimal);

  vk::SubpassDescription subpassDescription(
      {}, vk::PipelineBindPoint::eGraphics, 0, nullptr, colorAttachments.size(),
      colorAttachments.data(), nullptr, &depthAttachment);

  // ensure previous writes are done
  // TODO: compute a better dependency
  std::array<vk::SubpassDependency, 2> deps{
      vk::SubpassDependency(
          VK_SUBPASS_EXTERNAL, 0,
          vk::PipelineStageFlagBits::eColorAttachmentOutput |
              vk::PipelineStageFlagBits::eEarlyFragmentTests |
              vk::PipelineStageFlagBits::eLateFragmentTests,
          vk::PipelineStageFlagBits::eFragmentShader |
              vk::PipelineStageFlagBits::eColorAttachmentOutput,
          vk::AccessFlagBits::eColorAttachmentWrite |
              vk::AccessFlagBits::eDepthStencilAttachmentWrite,
          vk::AccessFlagBits::eShaderRead |
              vk::AccessFlagBits::eColorAttachmentWrite),
      vk::SubpassDependency(
          0, VK_SUBPASS_EXTERNAL,
          vk::PipelineStageFlagBits::eColorAttachmentOutput |
              vk::PipelineStageFlagBits::eEarlyFragmentTests |
              vk::PipelineStageFlagBits::eLateFragmentTests,
          vk::PipelineStageFlagBits::eFragmentShader |
              vk::PipelineStageFlagBits::eColorAttachmentOutput,
          vk::AccessFlagBits::eColorAttachmentWrite |
              vk::AccessFlagBits::eDepthStencilAttachmentWrite,
          vk::AccessFlagBits::eShaderRead |
              vk::AccessFlagBits::eColorAttachmentWrite),
  };

  mRenderPass = device.createRenderPassUnique(vk::RenderPassCreateInfo(
      {}, attachmentDescriptions.size(), attachmentDescriptions.data(), 1,
      &subpassDescription, 2, deps.data()));

  return mRenderPass.get();
}

vk::Pipeline GbufferPassParser::createGraphicsPipeline(
    vk::Device device, vk::Format colorFormat, vk::Format depthFormat,
    vk::CullModeFlags cullMode, vk::FrontFace frontFace,
    std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
        &colorTargetLayouts,
    std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
    std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts,
    std::map<std::string, SpecializationConstantValue> const
        &specializationConstantInfo) {
  // render pass
  auto renderPass = createRenderPass(device, colorFormat, depthFormat,
                                     colorTargetLayouts, depthLayout);

  // shaders
  vk::UniquePipelineCache pipelineCache =
      device.createPipelineCacheUnique(vk::PipelineCacheCreateInfo());
  auto vsm = device.createShaderModuleUnique(
      {{}, mVertSPVCode.size() * sizeof(uint32_t), mVertSPVCode.data()});
  auto fsm = device.createShaderModuleUnique(
      {{}, mFragSPVCode.size() * sizeof(uint32_t), mFragSPVCode.data()});

  auto elems = mSpecializationConstantLayout->getElementsSorted();
  vk::SpecializationInfo fragSpecializationInfo;
  std::vector<vk::SpecializationMapEntry> entries;
  std::vector<int> specializationData;
  if (elems.size()) {
    specializationData.resize(elems.size());
    for (uint32_t i = 0; i < elems.size(); ++i) {
      if (specializationConstantInfo.find(elems[i].name) !=
              specializationConstantInfo.end() &&
          elems[i].dtype !=
              specializationConstantInfo.at(elems[i].name).dtype) {
        throw std::runtime_error("Type mismatch on specialization constant " +
                                 elems[i].name + ".");
      }
      if (elems[i].dtype == eINT) {
        entries.emplace_back(elems[i].id, i * sizeof(int), sizeof(int));
        int v = specializationConstantInfo.find(elems[i].name) !=
                        specializationConstantInfo.end()
                    ? specializationConstantInfo.at(elems[i].name).intValue
                    : elems[i].intValue;
        std::memcpy(specializationData.data() + i, &v, sizeof(int));
      } else if (elems[i].dtype == eFLOAT) {
        entries.emplace_back(elems[i].id, i * sizeof(float), sizeof(float));
        float v = specializationConstantInfo.find(elems[i].name) !=
                          specializationConstantInfo.end()
                      ? specializationConstantInfo.at(elems[i].name).floatValue
                      : elems[i].floatValue;
        std::memcpy(specializationData.data() + i, &v, sizeof(float));
      } else {
        throw std::runtime_error(
            "only int and float are allowed specialization constants");
      }
    }
    fragSpecializationInfo = vk::SpecializationInfo(
        entries.size(), entries.data(), specializationData.size() * sizeof(int),
        specializationData.data());
  }

  std::array<vk::PipelineShaderStageCreateInfo, 2>
      pipelineShaderStageCreateInfos{
          vk::PipelineShaderStageCreateInfo(
              vk::PipelineShaderStageCreateFlags(),
              vk::ShaderStageFlagBits::eVertex, vsm.get(), "main", nullptr),
          vk::PipelineShaderStageCreateInfo(
              vk::PipelineShaderStageCreateFlags(),
              vk::ShaderStageFlagBits::eFragment, fsm.get(), "main",
              elems.size() ? &fragSpecializationInfo : nullptr)};

  // vertex input
  auto vertexInputBindingDescriptions =
      mVertexInputLayout->computeVertexInputBindingDescriptions();
  auto vertexInputAttributeDescriptions =
      mVertexInputLayout->computeVertexInputAttributesDescriptions();
  vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo;
  pipelineVertexInputStateCreateInfo.vertexBindingDescriptionCount =
      vertexInputBindingDescriptions.size();
  pipelineVertexInputStateCreateInfo.pVertexBindingDescriptions =
      vertexInputBindingDescriptions.data();
  pipelineVertexInputStateCreateInfo.vertexAttributeDescriptionCount =
      vertexInputAttributeDescriptions.size();
  pipelineVertexInputStateCreateInfo.pVertexAttributeDescriptions =
      vertexInputAttributeDescriptions.data();

  // input assembly
  vk::PipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo(
      vk::PipelineInputAssemblyStateCreateFlags(),
      vk::PrimitiveTopology::eTriangleList);

  // viewport
  vk::PipelineViewportStateCreateInfo pipelineViewportStateCreateInfo(
      vk::PipelineViewportStateCreateFlags(), 1, nullptr, 1, nullptr);

  // rasterization
  vk::PipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo(
      vk::PipelineRasterizationStateCreateFlags(), false, false,
      vk::PolygonMode::eFill, cullMode, frontFace, false, 0.0f, 0.0f, 0.0f,
      1.0f);

  // multisample
  vk::PipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo;

  // stencil
  vk::StencilOpState stencilOpState{};
  vk::PipelineDepthStencilStateCreateInfo pipelineDepthStencilStateCreateInfo(
      vk::PipelineDepthStencilStateCreateFlags(), true, true,
      vk::CompareOp::eLessOrEqual, false, false, stencilOpState,
      stencilOpState);

  // blend
  uint32_t numColorAttachments = mTextureOutputLayout->elements.size();
  vk::ColorComponentFlags colorComponentFlags(
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
  std::vector<vk::PipelineColorBlendAttachmentState>
      pipelineColorBlendAttachmentStates;

  auto outTextures = mTextureOutputLayout->getElementsSorted();
  for (uint32_t i = 0; i < numColorAttachments; ++i) {
    // alpha blend float textures
    if (mAlphaBlend && outTextures[i].dtype == DataType::eFLOAT4) {
      pipelineColorBlendAttachmentStates.push_back(
          vk::PipelineColorBlendAttachmentState(
              true, vk::BlendFactor::eSrcAlpha,
              vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd,
              vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
              colorComponentFlags));
    } else {
      pipelineColorBlendAttachmentStates.push_back(
          vk::PipelineColorBlendAttachmentState(
              false, vk::BlendFactor::eZero, vk::BlendFactor::eZero,
              vk::BlendOp::eAdd, vk::BlendFactor::eZero, vk::BlendFactor::eZero,
              vk::BlendOp::eAdd, colorComponentFlags));
    }
  }
  vk::PipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo(
      vk::PipelineColorBlendStateCreateFlags(), false, vk::LogicOp::eNoOp,
      numColorAttachments, pipelineColorBlendAttachmentStates.data(),
      {{1.0f, 1.0f, 1.0f, 1.0f}});

  // dynamic
  vk::DynamicState dynamicStates[2] = {vk::DynamicState::eViewport,
                                       vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo pipelineDynamicStateCreateInfo(
      vk::PipelineDynamicStateCreateFlags(), 2, dynamicStates);

  vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo(
      vk::PipelineCreateFlags(), pipelineShaderStageCreateInfos.size(),
      pipelineShaderStageCreateInfos.data(),
      &pipelineVertexInputStateCreateInfo,
      &pipelineInputAssemblyStateCreateInfo, nullptr,
      &pipelineViewportStateCreateInfo, &pipelineRasterizationStateCreateInfo,
      &pipelineMultisampleStateCreateInfo, &pipelineDepthStencilStateCreateInfo,
      &pipelineColorBlendStateCreateInfo, &pipelineDynamicStateCreateInfo,
      createPipelineLayout(device, descriptorSetLayouts), renderPass);
  mPipeline = device.createGraphicsPipelineUnique(pipelineCache.get(),
                                                  graphicsPipelineCreateInfo);
  return mPipeline.get();
}

std::vector<std::string> GbufferPassParser::getColorRenderTargetNames() const {
  std::vector<std::string> result;
  auto elems = mTextureOutputLayout->getElementsSorted();
  for (auto elem : elems) {
    result.push_back(getOutTextureName(elem.name));
  }
  return result;
}

std::optional<std::string> GbufferPassParser::getDepthRenderTargetName() const {
  return getName() + "Depth";
}

std::vector<UniformBindingType>
GbufferPassParser::getUniformBindingTypes() const {
  std::vector<UniformBindingType> result;
  for (auto &desc : mDescriptorSetDescriptions) {
    result.push_back(desc.type);
  }
  return result;
}

} // namespace shader
} // namespace svulkan2
