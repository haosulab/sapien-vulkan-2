#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace shader {
class DeferredPassParser : public BaseParser {

  std::shared_ptr<SpecializationConstantLayout> mSpecializationConstantLayout;
  std::shared_ptr<OutputDataLayout> mTextureOutputLayout;
  std::vector<DescriptorSetDescription> mDescriptorSetDescriptions;

public:
  inline std::shared_ptr<SpecializationConstantLayout>
  getSpecializationConstantLayout() const {
    return mSpecializationConstantLayout;
  }

  inline std::shared_ptr<OutputDataLayout>
  getTextureOutputLayout() const override {
    return mTextureOutputLayout;
  }

  vk::UniqueRenderPass createRenderPass(
      vk::Device device, std::vector<vk::Format> const &colorFormats,
      vk::Format depthFormat,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
          &colorTargetLayouts,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
      vk::SampleCountFlagBits sampleCount) const override;

  vk::UniquePipeline
  createPipeline(vk::Device device, vk::PipelineLayout layout,
                 vk::RenderPass renderPass, vk::CullModeFlags cullMode,
                 vk::FrontFace frontFace, bool alphaBlend,
                 vk::SampleCountFlagBits sampleCount,
                 std::map<std::string, SpecializationConstantValue> const
                     &specializationConstantInfo) const override;

  std::vector<std::string> getColorRenderTargetNames() const override;
  std::vector<std::string> getInputTextureNames() const override;
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
