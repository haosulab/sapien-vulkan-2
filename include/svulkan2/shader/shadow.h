#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace shader {

class ShadowPassParser : public BaseParser {
  std::shared_ptr<InputDataLayout> mVertexInputLayout;
  std::vector<DescriptorSetDescription> mDescriptorSetDescriptions;

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
