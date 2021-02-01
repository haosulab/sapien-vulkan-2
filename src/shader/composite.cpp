#include "svulkan2/shader/composite.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {
namespace shader {

    inline std::string getOutTextureName(std::string variableName) {// remove "out" prefix
    return variableName.substr(3, std::string::npos);
}

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
    for (auto& elem : mCombinedSamplerLayout->elements) {
        ASSERT(elem.second.name.substr(0, 7) == "sampler",
            "[frag]texture sampler variable must start with \"sampler\"");
    }
    
    // validate out textures
    for (auto& elem : mTextureOutputLayout->elements) {
        ASSERT(elem.second.name.substr(0, 3) == "out",
            "[frag]all out texture variables must start with \"out\"");
    }
}

vk::RenderPass CompositePassParser::createRenderPass(vk::Device device, vk::Format colorFormat, vk::Format depthFormat,
    std::unordered_map<std::string, vk::ImageLayout> renderTargetFinalLayouts) {
    std::vector<vk::AttachmentDescription> attachmentDescriptions;
    std::vector<vk::AttachmentReference> colorAttachments;

    auto elems = mTextureOutputLayout->getElementsSorted();
    for (uint32_t i = 0; i < elems.size(); ++i) {
        colorAttachments.push_back({ i, vk::ImageLayout::eColorAttachmentOptimal });
        vk::Format format;
        if (elems[i].dtype == eFLOAT4) {
            format = colorFormat;
        }
        else if (elems[i].dtype == eUINT4) {
            format = vk::Format::eR32G32B32A32Sfloat;
        }
        else {
            throw std::runtime_error(
                "only float4 and uint4 are allowed in output attachments");
        }
        attachmentDescriptions.push_back(vk::AttachmentDescription(
            {}, format, vk::SampleCountFlagBits::e1,
            vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eStore,// color attachment load and store op
            vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,// stencil load and store op
            vk::ImageLayout::eUndefined, // TODO : Compute initial layout
            renderTargetFinalLayouts[getOutTextureName(elems[i].name)]));
    }
    attachmentDescriptions.push_back(vk::AttachmentDescription(
        vk::AttachmentDescriptionFlags(), depthFormat,
        vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore,// depth attachment
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,// stencil
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eShaderReadOnlyOptimal)); // TODO: final layout should be
                                                   // computed
    vk::AttachmentReference depthAttachment(
        elems.size(), vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vk::SubpassDescription subpassDescription(
        {}, vk::PipelineBindPoint::eGraphics, 0, nullptr, colorAttachments.size(),
        colorAttachments.data(), nullptr, &depthAttachment);

    mRenderPass = device.createRenderPassUnique(vk::RenderPassCreateInfo(
        {}, attachmentDescriptions.size(), attachmentDescriptions.data(), 1,
        &subpassDescription));

    return mRenderPass.get();
}


vk::Pipeline CompositePassParser::createGraphicsPipeline(
    vk::Device device, vk::PipelineLayout pipelineLayout,
    vk::Format colorFormat, vk::Format depthFormat,
    std::unordered_map<std::string, vk::ImageLayout> renderTargetFinalLayouts) {
    // render pass
    auto renderPass = createRenderPass(device, colorFormat, depthFormat, renderTargetFinalLayouts);

    // shaders
    vk::UniquePipelineCache pipelineCache =
        device.createPipelineCacheUnique(vk::PipelineCacheCreateInfo());
    auto vsm = device.createShaderModuleUnique(
        { {}, mVertSPVCode.size(), mVertSPVCode.data() });
    auto fsm = device.createShaderModuleUnique(
        { {}, mFragSPVCode.size(), mFragSPVCode.data() });

    std::array<vk::PipelineShaderStageCreateInfo, 2>
        pipelineShaderStageCreateInfos{
            vk::PipelineShaderStageCreateInfo(
                vk::PipelineShaderStageCreateFlags(),
                vk::ShaderStageFlagBits::eVertex, vsm.get(), "main", nullptr),
            vk::PipelineShaderStageCreateInfo(
                vk::PipelineShaderStageCreateFlags(),
                vk::ShaderStageFlagBits::eFragment, fsm.get(), "main", nullptr) };

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
        vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise, false, 0.0f, 0.0f, 0.0f,
        1.0f);

    // multisample
    vk::PipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo;

    // stencil
    vk::StencilOpState stencilOpState{};
    vk::PipelineDepthStencilStateCreateInfo pipelineDepthStencilStateCreateInfo(
        vk::PipelineDepthStencilStateCreateFlags(),
        true, true, vk::CompareOp::eLessOrEqual, false,// depth test enabled , depth write enabled, depth compare op, depth Bounds Test Enable
        false, stencilOpState, stencilOpState);// stensil test enabled, stensil front, stensil back

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
        { {1.0f, 1.0f, 1.0f, 1.0f} });

    // dynamic
    vk::DynamicState dynamicStates[2] = { vk::DynamicState::eViewport,
                                         vk::DynamicState::eScissor };
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
        pipelineLayout, renderPass);
    mPipeline = device.createGraphicsPipelineUnique(pipelineCache.get(),
        graphicsPipelineCreateInfo);
    return mPipeline.get();
}


} // namespace shader
} // namespace svulkan2
