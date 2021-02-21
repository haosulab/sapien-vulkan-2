#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace shader {

class ShadowPassParser : public BaseParser {
  std::shared_ptr<InputDataLayout> mVertexInputLayout;
  // std::shared_ptr<StructDataLayout> mLightSpaceBufferLayout;
  // std::shared_ptr<StructDataLayout> mObjectBufferLayout;

  std::vector<DescriptorSetDescription> mDescriptorSetDescriptions;

  vk::UniqueRenderPass mRenderPass;
  vk::UniquePipeline mPipeline;

public:
  // Is this needed?
  // std::vector<std::string> getOutputTextureNames() const;

  inline std::shared_ptr<InputDataLayout> getVertexInputLayout() const {
    return mVertexInputLayout;
  }

  inline std::shared_ptr<OutputDataLayout>
  getTextureOutputLayout() const override {
    return nullptr;
  };

  inline std::vector<std::string> getColorRenderTargetNames() const override {
    return {};
  };
  inline std::optional<std::string> getDepthRenderTargetName() const override {
    return "ShadowDepth";
  };

  vk::PipelineLayout
  createPipelineLayout(vk::Device device,
                       std::vector<vk::DescriptorSetLayout> layouts);

  vk::RenderPass createRenderPass(
      vk::Device device, vk::Format depthFormat,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout);

  vk::Pipeline createGraphicsPipeline(
      vk::Device device, vk::Format colorFormat, vk::Format depthFormat,
      vk::CullModeFlags cullMode, vk::FrontFace frontFace,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
          &colorTargetLayouts,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
      std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts,
      std::map<std::string, SpecializationConstantValue> const
          &specializationConstantInfo) override;

  inline vk::RenderPass getRenderPass() const override {
    return mRenderPass.get();
  }
  inline vk::Pipeline getPipeline() const override { return mPipeline.get(); }
  std::vector<UniformBindingType> getUniformBindingTypes() const override;

  inline std::vector<DescriptorSetDescription>
  getDescriptorSetDescriptions() const override {
    return mDescriptorSetDescriptions;
  };

private:
  void reflectSPV() override;
  void validate() const;
};

} // namespace shader
} // namespace svulkan2
