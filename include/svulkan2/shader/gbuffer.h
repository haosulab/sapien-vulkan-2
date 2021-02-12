#pragma once
#include "base_parser.h"
#include "svulkan2/common/config.h"

namespace svulkan2 {
namespace shader {

class GbufferPassParser : public BaseParser {
  std::shared_ptr<InputDataLayout> mVertexInputLayout;
  std::shared_ptr<StructDataLayout> mCameraBufferLayout;
  std::shared_ptr<StructDataLayout> mObjectBufferLayout;

  std::shared_ptr<StructDataLayout> mMaterialBufferLayout;
  std::shared_ptr<CombinedSamplerLayout> mCombinedSamplerLayout;
  std::shared_ptr<OutputDataLayout> mTextureOutputLayout;

  vk::UniqueRenderPass mRenderPass;
  vk::UniquePipeline mPipeline;

public:
  inline std::shared_ptr<InputDataLayout> getVertexInputLayout() const {
    return mVertexInputLayout;
  }
  inline std::shared_ptr<StructDataLayout> getCameraBufferLayout() const {
    return mCameraBufferLayout;
  }
  inline std::shared_ptr<StructDataLayout> getObjectBufferLayout() const {
    return mObjectBufferLayout;
  }

  inline std::shared_ptr<StructDataLayout> getMaterialBufferLayout() const {
    return mMaterialBufferLayout;
  }
  inline std::shared_ptr<CombinedSamplerLayout>
  getCombinedSamplerLayout() const {
    return mCombinedSamplerLayout;
  }
  inline std::shared_ptr<OutputDataLayout>
  getTextureOutputLayout() const override {
    return mTextureOutputLayout;
  }
  ShaderConfig::MaterialPipeline getMaterialType() const;

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

  virtual inline vk::RenderPass getRenderPass() const override {
    return mRenderPass.get();
  }
  virtual inline vk::Pipeline getPipeline() const override {
    return mPipeline.get();
  }
  virtual std::vector<std::string> getRenderTargetNames() const override;

private:
  void reflectSPV() override;
  void validate() const;
};

} // namespace shader
} // namespace svulkan2
