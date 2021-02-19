#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace shader {

class CompositePassParser : public BaseParser {
  std::shared_ptr<CombinedSamplerLayout> mCombinedSamplerLayout;
  std::shared_ptr<OutputDataLayout> mTextureOutputLayout;

  vk::UniqueRenderPass mRenderPass;
  vk::UniquePipeline mPipeline;

public:
  inline std::shared_ptr<CombinedSamplerLayout>
  getCombinedSamplerLayout() const {
    return mCombinedSamplerLayout;
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
      vk::Device device, vk::Format colorFormat,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const &layouts);

  vk::Pipeline createGraphicsPipeline(
      vk::Device device, vk::Format colorFormat,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
          &renderTargetLayouts,
      std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts);

  virtual std::vector<std::string> getRenderTargetNames() const override;
  std::vector<std::string> getInputTextureNames() const override;
  std::vector<UniformBindingType> getUniformBindingTypes() const override;

private:
  void reflectSPV() override;
  void validate() const;
};

} // namespace shader
} // namespace svulkan2
