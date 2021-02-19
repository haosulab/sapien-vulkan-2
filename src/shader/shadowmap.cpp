#include "svulkan2/shader/shadowmap.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {
namespace shader {

void ShadowPassParser::reflectSPV() {
  spirv_cross::Compiler vertComp(mVertSPVCode);
  try {
    mVertexInputLayout = parseVertexInput(vertComp);
    mLightSpaceBufferLayout = parseLightSpaceBuffer(vertComp, 0, 0);
    mObjectBufferLayout = parseObjectBuffer(vertComp, 0, 1);
  } catch (std::runtime_error const &err) {
    throw std::runtime_error("[vert]" + std::string(err.what()));
  }
}

void ShadowPassParser::validate() const {
  // Is this needed any more?
}

vk::PipelineLayout ShadowPassParser::createPipelineLayout(
    vk::Device device,
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts) {
  vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
  pipelineLayoutInfo.setLayoutCount = descriptorSetLayouts.size();
  pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
  mPipelineLayout = device.createPipelineLayoutUnique(pipelineLayoutInfo);

  return mPipelineLayout.get();
}

vk::RenderPass ShadowPassParser::createRenderPass(
    vk::Device device, vk::Format depthFormat,
    std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout) {
  std::vector<vk::AttachmentDescription> attachmentDescriptions;

  // Only Depth Attachment needed for Shadow Mapping
  attachmentDescriptions.push_back(vk::AttachmentDescription(
      vk::AttachmentDescriptionFlags(), depthFormat,
      vk::SampleCountFlagBits::e1,
      depthLayout.first == vk::ImageLayout::eUndefined
          ? vk::AttachmentLoadOp::eClear
          : vk::AttachmentLoadOp::eLoad,
      vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
      vk::AttachmentStoreOp::eDontCare,
      depthLayout.first,    // Initial Layout
      depthLayout.second)); // Final Layout

  vk::AttachmentReference depthAttachment(
      0, vk::ImageLayout::eDepthStencilAttachmentOptimal);

  vk::SubpassDescription subpassDescription(
      {}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 0, nullptr, nullptr,
      &depthAttachment);

  mRenderPass = device.createRenderPassUnique(vk::RenderPassCreateInfo(
      {}, attachmentDescriptions.size(), attachmentDescriptions.data(), 1,
      &subpassDescription));

  return mRenderPass.get();
}

vk::Pipeline ShadowPassParser::createGraphicsPipeline(
    vk::Device device, vk::Format depthFormat, vk::CullModeFlags cullMode,
    vk::FrontFace frontFace,
    std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
    std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts) {
  // render pass
  auto renderPass = createRenderPass(device, depthFormat, depthLayout);

  // shaders
  vk::UniquePipelineCache pipelineCache =
      device.createPipelineCacheUnique(vk::PipelineCacheCreateInfo());
  auto vsm = device.createShaderModuleUnique(
      {{}, mVertSPVCode.size() * sizeof(uint32_t), mVertSPVCode.data()});

  std::array<vk::PipelineShaderStageCreateInfo, 1>
      pipelineShaderStageCreateInfos{vk::PipelineShaderStageCreateInfo(
          vk::PipelineShaderStageCreateFlags(),
          vk::ShaderStageFlagBits::eVertex, vsm.get(), "main", nullptr)};

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
      nullptr, &pipelineDynamicStateCreateInfo,
      createPipelineLayout(device, descriptorSetLayouts), renderPass);
  mPipeline = device
                  .createGraphicsPipelineUnique(pipelineCache.get(),
                                                graphicsPipelineCreateInfo)
                  .value;
  return mPipeline.get();
}

std::vector<UniformBindingType>
ShadowPassParser::getUniformBindingTypes() const {
  return {UniformBindingType::eLight, UniformBindingType::eObject};
}

} // namespace shader
} // namespace svulkan2
