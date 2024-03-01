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

public:
  inline std::shared_ptr<InputDataLayout> getVertexInputLayout() const {
    return mVertexInputLayout;
  }

  inline std::shared_ptr<OutputDataLayout> getTextureOutputLayout() const override {
    return mTextureOutputLayout;
  }

  vk::UniqueRenderPass createRenderPass(
      vk::Device device, std::vector<vk::Format> const &colorFormats, vk::Format depthFormat,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const &colorTargetLayouts,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
      vk::SampleCountFlagBits sampleCount) const override;

  std::vector<std::string> getColorRenderTargetNames() const override;
  std::optional<std::string> getDepthRenderTargetName() const override;
  std::vector<UniformBindingType> getUniformBindingTypes() const override;

  inline std::vector<DescriptorSetDescription> getDescriptorSetDescriptions() const override {
    return mDescriptorSetDescriptions;
  };

private:
  void reflectSPV() override;
  void validate() const;

protected:
  // primitiveType 0 for point, 1 for line
  vk::UniquePipeline createPipelineHelper(
      vk::Device device, vk::PipelineLayout layout, vk::RenderPass renderPass,
      vk::CullModeFlags cullMode, vk::FrontFace frontFace, bool alphaBlend,
      vk::SampleCountFlagBits sampleCount,
      std::map<std::string, SpecializationConstantValue> const &specializationConstantInfo,
      int primitiveType) const;
};

class LinePassParser : public PrimitivePassParser {

public:
  vk::UniquePipeline createPipeline(vk::Device device, vk::PipelineLayout layout,
                                    vk::RenderPass renderPass, vk::CullModeFlags cullMode,
                                    vk::FrontFace frontFace, bool alphaBlend,
                                    vk::SampleCountFlagBits sampleCount,
                                    std::map<std::string, SpecializationConstantValue> const
                                        &specializationConstantInfo) const override;
};

class PointPassParser : public PrimitivePassParser {

public:
  vk::UniquePipeline createPipeline(vk::Device device, vk::PipelineLayout layout,
                                    vk::RenderPass renderPass, vk::CullModeFlags cullMode,
                                    vk::FrontFace frontFace, bool alphaBlend,
                                    vk::SampleCountFlagBits sampleCount,
                                    std::map<std::string, SpecializationConstantValue> const
                                        &specializationConstantInfo) const override;
};

} // namespace shader
} // namespace svulkan2
