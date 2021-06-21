#pragma once
#include "base_parser.h"
#include "svulkan2/common/config.h"
namespace svulkan2 {
namespace shader {

class PrimitivePassParser : public BaseParser {
  std::shared_ptr<SpecializationConstantLayout> mSpecializationConstantLayout;
  std::shared_ptr<InputDataLayout> mVertexInputLayout;
  std::shared_ptr<OutputDataLayout> mTextureOutputLayout;
  std::vector<DescriptorSetDescription> mDescriptorSetDescriptions;

  vk::UniqueRenderPass mRenderPass;
  vk::UniquePipeline mPipeline;

public:
  inline std::shared_ptr<InputDataLayout> getVertexInputLayout() const {
    return mVertexInputLayout;
  }

  inline std::shared_ptr<OutputDataLayout>
  getTextureOutputLayout() const override {
    return mTextureOutputLayout;
  }

  vk::PipelineLayout
  createPipelineLayout(vk::Device device,
                       std::vector<vk::DescriptorSetLayout> layouts);

  vk::RenderPass createRenderPass(
      vk::Device device, vk::Format colorFormat, vk::Format depthFormat,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
          &colorTargetLayouts,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout);

  inline vk::RenderPass getRenderPass() const override {
    return mRenderPass.get();
  }

  inline vk::Pipeline getPipeline() const override { return mPipeline.get(); }
  std::vector<std::string> getColorRenderTargetNames() const override;
  std::optional<std::string> getDepthRenderTargetName() const override;
  std::vector<UniformBindingType> getUniformBindingTypes() const override;

  inline std::vector<DescriptorSetDescription>
  getDescriptorSetDescriptions() const override {
    return mDescriptorSetDescriptions;
  };

private:
  void reflectSPV() override;
  void validate() const;

protected:
  // primitiveType 0 for point, 1 for line
  vk::Pipeline createGraphicsPipelineHelper(
      vk::Device device, vk::Format colorFormat, vk::Format depthFormat,
      vk::CullModeFlags cullMode, vk::FrontFace frontFace,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
          &colorTargetLayouts,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
      std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts,
      std::map<std::string, SpecializationConstantValue> const
          &specializationConstantInfo,
      int primitiveType, float primitiveSize);
};

class LinePassParser : public PrimitivePassParser {

  float mLineWidth{1.f};

public:
  inline void setLineWidth(float w) { mLineWidth = w; }

  inline vk::Pipeline createGraphicsPipeline(
      vk::Device device, vk::Format colorFormat, vk::Format depthFormat,
      vk::CullModeFlags cullMode, vk::FrontFace frontFace,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
          &colorTargetLayouts,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
      std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts,
      std::map<std::string, SpecializationConstantValue> const
          &specializationConstantInfo) override {
    return createGraphicsPipelineHelper(
        device, colorFormat, depthFormat, cullMode, frontFace,
        colorTargetLayouts, depthLayout, descriptorSetLayouts,
        specializationConstantInfo, 1, mLineWidth);
  }
};

class PointPassParser : public PrimitivePassParser {

  float mLineWidth{1.f};

public:
  inline void setLineWidth(float w) { mLineWidth = w; }

  inline vk::Pipeline createGraphicsPipeline(
      vk::Device device, vk::Format colorFormat, vk::Format depthFormat,
      vk::CullModeFlags cullMode, vk::FrontFace frontFace,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
          &colorTargetLayouts,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
      std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts,
      std::map<std::string, SpecializationConstantValue> const
          &specializationConstantInfo) override {
    return createGraphicsPipelineHelper(device, colorFormat, depthFormat,
                                        cullMode, frontFace, colorTargetLayouts,
                                        depthLayout, descriptorSetLayouts,
                                        specializationConstantInfo, 0, 0);
  }
};

} // namespace shader
} // namespace svulkan2
