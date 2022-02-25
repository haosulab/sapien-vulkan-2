#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace shader {

class PrimitiveShadowPassParser : public BaseParser {
  std::shared_ptr<InputDataLayout> mVertexInputLayout;
  std::vector<DescriptorSetDescription> mDescriptorSetDescriptions;

  vk::UniqueRenderPass mRenderPass;
  vk::UniquePipeline mPipeline;

public:
  inline std::shared_ptr<InputDataLayout> getVertexInputLayout() const {
    return mVertexInputLayout;
  }

  inline std::shared_ptr<OutputDataLayout>
  getTextureOutputLayout() const override {
    return std::make_shared<OutputDataLayout>();
  };

  inline std::vector<std::string> getColorRenderTargetNames() const override {
    return {};
  };
  std::optional<std::string> getDepthRenderTargetName() const override;

  vk::PipelineLayout
  createPipelineLayout(vk::Device device,
                       std::vector<vk::DescriptorSetLayout> layouts);

  vk::RenderPass createRenderPass(
      vk::Device device, vk::Format depthFormat,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout);

  inline vk::RenderPass getRenderPass() const override {
    return mRenderPass.get();
  }
  inline vk::Pipeline getPipeline() const override { return mPipeline.get(); }
  std::vector<UniformBindingType> getUniformBindingTypes() const override;

  inline std::vector<DescriptorSetDescription>
  getDescriptorSetDescriptions() const override {
    return mDescriptorSetDescriptions;
  };

protected:
  vk::Pipeline createGraphicsPipelineHelper(
      vk::Device device, std::vector<vk::Format> const &colorFormats,
      vk::Format depthFormat, vk::CullModeFlags cullMode,
      vk::FrontFace frontFace,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
          &colorTargetLayouts,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
      std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts,
      std::map<std::string, SpecializationConstantValue> const
          &specializationConstantInfo,
      int primitiveType, float primitiveSize);

private:
  void reflectSPV() override;
  void validate() const;
};

class PointShadowParser : public PrimitiveShadowPassParser {
public:
  vk::Pipeline createGraphicsPipeline(
      vk::Device device, std::vector<vk::Format> const &colorFormats,
      vk::Format depthFormat, vk::CullModeFlags cullMode,
      vk::FrontFace frontFace,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
          &colorTargetLayouts,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
      std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts,
      std::map<std::string, SpecializationConstantValue> const
          &specializationConstantInfo) {
    return createGraphicsPipelineHelper(device, colorFormats, depthFormat,
                                        cullMode, frontFace, colorTargetLayouts,
                                        depthLayout, descriptorSetLayouts,
                                        specializationConstantInfo, 0, 0.f);
  }
};

} // namespace shader
} // namespace svulkan2
