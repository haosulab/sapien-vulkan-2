#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace shader {

class ShadowPassParser : public BaseParser {
  std::shared_ptr<InputDataLayout> mVertexInputLayout;
  std::shared_ptr<StructDataLayout> mLightSpaceBufferLayout;
  std::shared_ptr<StructDataLayout> mObjectBufferLayout;

  vk::UniqueRenderPass mRenderPass;
  vk::UniquePipeline mPipeline;

public:
  // Is this needed?
  // std::vector<std::string> getOutputTextureNames() const;

  inline std::shared_ptr<InputDataLayout> getVertexInputLayout() const {
    return mVertexInputLayout;
  }
  inline std::shared_ptr<StructDataLayout> getLightSpaceBufferLayout() const {
    return mLightSpaceBufferLayout;
  }
  inline std::shared_ptr<StructDataLayout> getObjectBufferLayout() const {
    return mObjectBufferLayout;
  }

  inline std::shared_ptr<OutputDataLayout>
  getTextureOutputLayout() const override {
    return nullptr;
  };

  inline std::vector<std::string> getRenderTargetNames() const override {
    return {"ShadowDepthMap"};
  };

  vk::PipelineLayout
  createPipelineLayout(vk::Device device,
                       std::vector<vk::DescriptorSetLayout> layouts);

  vk::RenderPass createRenderPass(
      vk::Device device, vk::Format depthFormat,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout);

  vk::Pipeline createGraphicsPipeline(
      vk::Device device, vk::Format depthFormat, vk::CullModeFlags cullMode,
      vk::FrontFace frontFace,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
      std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts);

  inline vk::RenderPass getRenderPass() const override {
    return mRenderPass.get();
  }
  inline vk::Pipeline getPipeline() const override { return mPipeline.get(); }
  std::vector<UniformBindingType> getUniformBindingTypes() const override;

private:
  void reflectSPV() override;
  void validate() const;
};

} // namespace shader
} // namespace svulkan2
