#include "svulkan2/shader/deferred.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {
namespace shader {

inline std::string getOutTextureName(std::string variableName) {// remove "out" prefix
    return variableName.substr(3, std::string::npos);
}

void DeferredPassParser::reflectSPV() {
  spirv_cross::Compiler vertComp(mVertSPVCode);
  spirv_cross::Compiler fragComp(mFragSPVCode);
  try {
    mSpecializationConstantLayout = parseSpecializationConstant(fragComp);
    mCameraBufferLayout = parseCameraBuffer(fragComp, 0, 1);
    mSceneBufferLayout = parseSceneBuffer(fragComp, 0, 0);
    mCombinedSamplerLayout = parseCombinedSampler(fragComp);
    mTextureOutputLayout = parseTextureOutput(fragComp);
  } catch (std::runtime_error const &err) {
    throw std::runtime_error("[frag]" + std::string(err.what()));
  }

  validate();

}

void DeferredPassParser::validate() const {
  // validate constants
  ASSERT(CONTAINS(mSpecializationConstantLayout->elements,
                  "NUM_DIRECTIONAL_LIGHTS"),
         "[frag]NUM_DIRECTIONAL_LIGHTS is a required specialization "
         "constant");
  ASSERT(CONTAINS(mSpecializationConstantLayout->elements, "NUM_POINT_LIGHTS"),
         "[frag]NUM_POINT_LIGHTS is a required specialization "
         "constant");

  // validate samplers
  for (auto &sampler : mCombinedSamplerLayout->elements) {
    ASSERT(sampler.second.name.length() > 7 &&
               sampler.second.name.substr(0, 7) == "sampler",
           "[frag]texture sampler variable must start with \"sampler\"");
    ASSERT(sampler.second.set == 2, "all deferred.frag: all texture sampler "
                                    "should be bound at descriptor set 2");
  }

  // validate out textures
  for (auto& elem : mTextureOutputLayout->elements) {
      ASSERT(elem.second.name.substr(0, 3) == "out",
          "[frag]all out texture variables must start with \"out\"");
  }
}

vk::PipelineLayout DeferredPassParser::createPipelineLayout(vk::Device device) {
    //process gbuffer uniforms and samplers:
    int numGbufferSamplers = mCombinedSamplerLayout->elements.size();
    std::vector<vk::DescriptorSetLayoutBinding> gBufferBindings(2 + numGbufferSamplers);// Magic number 2 for Camera, scene bufers.
    //scene buffer(set 0, binding 1), camera buffer(set 1, binding 0)
    for (int i = 0; i < 2; i++) {
        gBufferBindings[i].binding = 0;
        gBufferBindings[i].descriptorType = vk::DescriptorType::eUniformBuffer;
        gBufferBindings[i].descriptorCount = 1;
        gBufferBindings[i].stageFlags = vk::ShaderStageFlagBits::eFragment;
        gBufferBindings[i].pImmutableSamplers = nullptr;
    }
    //samplers(set 2):
    int i = 0;
    for (const auto elem : mCombinedSamplerLayout->elements) {
        //material buffer:
        gBufferBindings[2 + i].binding = elem.second.binding;
        gBufferBindings[2 + i].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        gBufferBindings[2 + i].descriptorCount = 1;
        gBufferBindings[2 + i].stageFlags = vk::ShaderStageFlagBits::eFragment;
        gBufferBindings[2 + i].pImmutableSamplers = nullptr;
        i++;
    }
    std::vector<vk::DescriptorSetLayoutCreateInfo> createInfo(3);
    createInfo[0].bindingCount = 1;
    createInfo[0].pBindings = gBufferBindings.data();
    createInfo[1].bindingCount = 1;
    createInfo[1].pBindings = gBufferBindings.data() + 1;
    createInfo[2].bindingCount = gBufferBindings.size() - 2;
    createInfo[2].pBindings = gBufferBindings.data() + 2;
    std::vector<vk::DescriptorSetLayout> dsLayout(3);// 3 descriptor set layouts for set = 0,1,2
    dsLayout[0] = device.createDescriptorSetLayout(createInfo[0]);
    dsLayout[1] = device.createDescriptorSetLayout(createInfo[1]);
    dsLayout[2] = device.createDescriptorSetLayout(createInfo[2]);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setLayoutCount = dsLayout.size();
    pipelineLayoutInfo.pSetLayouts = dsLayout.data();
    mPipelineLayout = device.createPipelineLayoutUnique(pipelineLayoutInfo);

    return mPipelineLayout.get();
}


vk::RenderPass DeferredPassParser::createRenderPass(vk::Device device, vk::Format colorFormat, vk::Format depthFormat,
    std::unordered_map<std::string, std::pair<vk::ImageLayout, vk::ImageLayout>> layouts) {
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
            layouts[getOutTextureName(elems[i].name)].first,
            layouts[getOutTextureName(elems[i].name)].second));
    }
    attachmentDescriptions.push_back(vk::AttachmentDescription(
        vk::AttachmentDescriptionFlags(), depthFormat,
        vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore,// depth attachment
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,// stencil
        vk::ImageLayout::eDepthStencilReadOnlyOptimal,
        vk::ImageLayout::eDepthStencilReadOnlyOptimal));

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


vk::Pipeline DeferredPassParser::createGraphicsPipeline(
    vk::Device device,
    vk::Format colorFormat, vk::Format depthFormat,
    std::unordered_map<std::string, std::pair<vk::ImageLayout, vk::ImageLayout>> renderTargetLayouts,
    int numDirectionalLights, int numPointLights) {
    // render pass
    auto renderPass = createRenderPass(device, colorFormat, depthFormat, renderTargetLayouts);

    // shaders
    vk::UniquePipelineCache pipelineCache =
        device.createPipelineCacheUnique(vk::PipelineCacheCreateInfo());
    auto vsm = device.createShaderModuleUnique(
        { {}, mVertSPVCode.size(), mVertSPVCode.data() });
    auto fsm = device.createShaderModuleUnique(
        { {}, mFragSPVCode.size(), mFragSPVCode.data() });
    
    //specialization Constants
    auto elems = mSpecializationConstantLayout->getElementsSorted();
    std::vector<vk::SpecializationMapEntry> entries;
    std::vector<int> specializationData;
    for (uint32_t i = 0; i < elems.size(); ++i) {
        if (elems[i].dtype == eINT) {
            entries.emplace_back(elems[i].id, i * sizeof(int), sizeof(int));
            if(elems[i].name == "NUM_DIRECTIONAL_LIGHTS" && numDirectionalLights != -1)
                specializationData.push_back(numDirectionalLights);
            else if (elems[i].name == "NUM_POINT_LIGHTS" && numPointLights != -1)
                specializationData.push_back(numPointLights);
            else
                specializationData.push_back(elems[i].intValue);
        }
        else {
            throw std::runtime_error(
                "only int is allowed for number of point and directional lights");
        }
    }
    vk::SpecializationInfo fragSpecializationInfo(entries.size(), entries.data(), specializationData.size() * sizeof(int), specializationData.data());

    std::array<vk::PipelineShaderStageCreateInfo, 2>
        pipelineShaderStageCreateInfos{
            vk::PipelineShaderStageCreateInfo(
                vk::PipelineShaderStageCreateFlags(),
                vk::ShaderStageFlagBits::eVertex, vsm.get(), "main", nullptr),
            vk::PipelineShaderStageCreateInfo(
                vk::PipelineShaderStageCreateFlags(),
                vk::ShaderStageFlagBits::eFragment, fsm.get(), "main", &fragSpecializationInfo) };

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
        createPipelineLayout(device), renderPass);
    mPipeline = device.createGraphicsPipelineUnique(pipelineCache.get(),
        graphicsPipelineCreateInfo);
    return mPipeline.get();
}

} // namespace shader
} // namespace svulkan2
