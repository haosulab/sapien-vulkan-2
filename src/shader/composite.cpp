#include "svulkan2/shader/composite.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {
namespace shader {

void CompositePassParser::reflectSPV() {
  spirv_cross::Compiler vertComp(mVertSPVCode);
  spirv_cross::Compiler fragComp(mFragSPVCode);
  try {
    mCombinedSamplerLayout = parseCombinedSampler(fragComp);
    mTextureOutputLayout = parseTextureOutput(fragComp);
  } catch (std::runtime_error const &err) {
    throw std::runtime_error("[frag]" + std::string(err.what()));
  }
}

void CompositePassParser::validate() const {
  // validate sampler
  for (auto &elem : mCombinedSamplerLayout->elements) {
    ASSERT(elem.second.name.substr(0, 7) == "sampler",
           "[frag]texture sampler variable must start with \"sampler\"");
  }

  // validate out textures
  for (auto &elem : mTextureOutputLayout->elements) {
    ASSERT(elem.second.name.substr(0, 3) == "out",
           "[frag]all out texture variables must start with \"out\"");
  }
}

vk::PipelineLayout CompositePassParser::createPipelineLayout(
    vk::Device device, std::vector<vk::DescriptorSetLayout> layouts) {
  // // process gbuffer uniforms and samplers:
  // int numGbufferSamplers = mCombinedSamplerLayout->elements.size();
  // std::vector<vk::DescriptorSetLayoutBinding> gBufferBindings(
  //     numGbufferSamplers);
  // // samplers(set 0):
  // int i = 0;
  // for (const auto &elem : mCombinedSamplerLayout->elements) {
  //   // material buffer:
  //   gBufferBindings[i].binding = elem.second.binding;
  //   gBufferBindings[i].descriptorType =
  //       vk::DescriptorType::eCombinedImageSampler;
  //   gBufferBindings[i].descriptorCount = 1;
  //   gBufferBindings[i].stageFlags = vk::ShaderStageFlagBits::eFragment;
  //   gBufferBindings[i].pImmutableSamplers = nullptr;
  //   i++;
  // }
  // vk::DescriptorSetLayoutCreateInfo createInfo;
  // createInfo.bindingCount = gBufferBindings.size();
  // createInfo.pBindings = gBufferBindings.data();
  // vk::DescriptorSetLayout dsLayout;
  // dsLayout = device.createDescriptorSetLayout(createInfo);

  vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, layouts.size(),
                                                  layouts.data());
  mPipelineLayout = device.createPipelineLayoutUnique(pipelineLayoutInfo);

  return mPipelineLayout.get();
}

vk::RenderPass CompositePassParser::createRenderPass(
    vk::Device device, vk::Format colorFormat,
    std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const &layouts) {
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
        vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eStore, // color attachment load and store op
        vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eDontCare, // stencil load and store op
        layouts[i].first, layouts[i].second));
  }

  vk::SubpassDescription subpassDescription(
      {}, vk::PipelineBindPoint::eGraphics, 0, nullptr, colorAttachments.size(),
      colorAttachments.data(), nullptr, nullptr);

  mRenderPass = device.createRenderPassUnique(vk::RenderPassCreateInfo(
      {}, attachmentDescriptions.size(), attachmentDescriptions.data(), 1,
      &subpassDescription));

  return mRenderPass.get();
}

vk::Pipeline CompositePassParser::createGraphicsPipeline(
    vk::Device device, vk::Format colorFormat,
    std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
        &renderTargetLayouts,
    std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts) {
  // render pass
  auto renderPass = createRenderPass(device, colorFormat, renderTargetLayouts);

  // shaders
  vk::UniquePipelineCache pipelineCache =
      device.createPipelineCacheUnique(vk::PipelineCacheCreateInfo());
  auto vsm = device.createShaderModuleUnique(
      {{}, mVertSPVCode.size() * sizeof(uint32_t), mVertSPVCode.data()});
  auto fsm = device.createShaderModuleUnique(
      {{}, mFragSPVCode.size() * sizeof(uint32_t), mFragSPVCode.data()});

  std::array<vk::PipelineShaderStageCreateInfo, 2>
      pipelineShaderStageCreateInfos{
          vk::PipelineShaderStageCreateInfo(
              vk::PipelineShaderStageCreateFlags(),
              vk::ShaderStageFlagBits::eVertex, vsm.get(), "main", nullptr),
          vk::PipelineShaderStageCreateInfo(
              vk::PipelineShaderStageCreateFlags(),
              vk::ShaderStageFlagBits::eFragment, fsm.get(), "main", nullptr)};

  // vertex input
  // drawing a single hardcoded triangle
  vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo;
  pipelineVertexInputStateCreateInfo.vertexBindingDescriptionCount = 0;
  pipelineVertexInputStateCreateInfo.pVertexBindingDescriptions = nullptr;
  pipelineVertexInputStateCreateInfo.vertexAttributeDescriptionCount = 0;
  pipelineVertexInputStateCreateInfo.pVertexAttributeDescriptions = nullptr;

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
      vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
      vk::FrontFace::eCounterClockwise, false, 0.0f, 0.0f, 0.0f, 1.0f);

  // multisample
  vk::PipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo;

  // stencil
  vk::StencilOpState stencilOpState{};
  vk::PipelineDepthStencilStateCreateInfo pipelineDepthStencilStateCreateInfo(
      vk::PipelineDepthStencilStateCreateFlags(), true, true,
      vk::CompareOp::eLessOrEqual,
      false, // depth test enabled , depth write enabled, depth compare op,
             // depth Bounds Test Enable
      false, stencilOpState,
      stencilOpState); // stensil test enabled, stensil front, stensil back

  // blend
  uint32_t numColorAttachments = mTextureOutputLayout->elements.size();
  vk::ColorComponentFlags colorComponentFlags(
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
  std::vector<vk::PipelineColorBlendAttachmentState>
      pipelineColorBlendAttachmentStates;
  for (uint32_t i = 0; i < numColorAttachments; ++i) {
    pipelineColorBlendAttachmentStates.push_back(
        vk::PipelineColorBlendAttachmentState(
            false, vk::BlendFactor::eZero, vk::BlendFactor::eZero,
            vk::BlendOp::eAdd, vk::BlendFactor::eZero, vk::BlendFactor::eZero,
            vk::BlendOp::eAdd, colorComponentFlags));
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

std::vector<std::string> CompositePassParser::getRenderTargetNames() const {
  std::vector<std::string> result;
  auto elems = mTextureOutputLayout->getElementsSorted();
  for (auto elem : elems) {
    result.push_back(getOutTextureName(elem.name));
  }
  return result;
}

} // namespace shader
} // namespace svulkan2
