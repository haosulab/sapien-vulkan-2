#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace shader {
class DeferredPassParser : public BaseParser {

  std::shared_ptr<SpecializationConstantLayout> mSpecializationConstantLayout;
  std::shared_ptr<OutputDataLayout> mTextureOutputLayout;
  std::vector<DescriptorSetDescription> mDescriptorSetDescriptions;

  vk::UniqueRenderPass mRenderPass;
  vk::UniquePipeline mPipeline;

public:
  inline std::shared_ptr<SpecializationConstantLayout>
  getSpecializationConstantLayout() const {
    return mSpecializationConstantLayout;
  }

  inline std::shared_ptr<OutputDataLayout>
  getTextureOutputLayout() const override {
    return mTextureOutputLayout;
  }

  virtual inline vk::RenderPass getRenderPass() const override {
    return mRenderPass.get();
  }
  virtual inline vk::Pipeline getPipeline() const override {
    return mPipeline.get();
  }

  vk::PipelineLayout
  createPipelineLayout(vk::Device device,
                       std::vector<vk::DescriptorSetLayout> layouts);

  vk::RenderPass createRenderPass(
      vk::Device device, std::vector<vk::Format> const &colorFormats,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const &layouts);

  virtual vk::Pipeline createGraphicsPipeline(
      vk::Device device, std::vector<vk::Format> const &colorFormats,
      vk::Format depthFormat, vk::CullModeFlags cullMode,
      vk::FrontFace frontFace,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
          &colorTargetLayouts,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
      std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts,
      std::map<std::string, SpecializationConstantValue> const
          &specializationConstantInfo) override;

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
