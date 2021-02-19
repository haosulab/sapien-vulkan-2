#pragma once
#include "base_parser.h"
#include "svulkan2/common/config.h"

namespace svulkan2 {
namespace shader {

class GbufferPassParser : public BaseParser {
  std::shared_ptr<InputDataLayout> mVertexInputLayout;
  std::shared_ptr<OutputDataLayout> mTextureOutputLayout;
  // std::shared_ptr<StructDataLayout> mCameraBufferLayout;
  // std::shared_ptr<StructDataLayout> mObjectBufferLayout;

  // std::shared_ptr<StructDataLayout> mMaterialBufferLayout;
  // std::shared_ptr<CombinedSamplerLayout> mCombinedSamplerLayout;
  std::vector<DescriptorSetDescription> mDescriptorSetDescriptions;

  vk::UniqueRenderPass mRenderPass;
  vk::UniquePipeline mPipeline;

public:
  inline std::shared_ptr<InputDataLayout> getVertexInputLayout() const {
    return mVertexInputLayout;
  }
  // inline std::shared_ptr<StructDataLayout> getCameraBufferLayout() const {
  //   return mCameraBufferLayout;
  // }
  // inline std::shared_ptr<StructDataLayout> getObjectBufferLayout() const {
  //   return mObjectBufferLayout;
  // }

  // inline std::shared_ptr<StructDataLayout> getMaterialBufferLayout() const {
  //   return mMaterialBufferLayout;
  // }
  // inline std::shared_ptr<CombinedSamplerLayout>
  // getCombinedSamplerLayout() const {
  //   return mCombinedSamplerLayout;
  // }
  // ShaderConfig::MaterialPipeline getMaterialType() const;

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

  vk::Pipeline createGraphicsPipeline(
      vk::Device device, vk::Format colorFormat, vk::Format depthFormat,
      vk::CullModeFlags cullMode, vk::FrontFace frontFace,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
          &colorTargetLayouts,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
      std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts);

  inline vk::RenderPass getRenderPass() const override {
    return mRenderPass.get();
  }
  inline vk::Pipeline getPipeline() const override { return mPipeline.get(); }
  std::vector<std::string> getRenderTargetNames() const override;
  std::vector<UniformBindingType> getUniformBindingTypes() const override;

  inline std::vector<DescriptorSetDescription>
  getDescriptorSetDescriptions() const {
    return mDescriptorSetDescriptions;
  };

private:
  void reflectSPV() override;
  void validate() const;
};

} // namespace shader
} // namespace svulkan2
