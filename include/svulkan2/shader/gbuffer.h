#pragma once
#include "base_parser.h"

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
  enum { eSPECULAR, eMETALLIC } mMaterialType;

  std::vector<std::string> getOutputTextureNames() const;

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
  inline std::shared_ptr<OutputDataLayout> getTextureOutputLayout() const {
    return mTextureOutputLayout;
  }

  vk::PipelineLayout createPipelineLayout(vk::Device device) override;

  vk::RenderPass createRenderPass(vk::Device device, vk::Format colorFormat, vk::Format depthFormat,
      std::unordered_map<std::string, std::pair<vk::ImageLayout, vk::ImageLayout>> layouts);

  vk::Pipeline
  createGraphicsPipeline(vk::Device device,
                         vk::Format colorFormat, vk::Format depthFormat,
                         vk::CullModeFlags cullMode, vk::FrontFace frontFace, std::unordered_map<std::string, std::pair<vk::ImageLayout, vk::ImageLayout>> renderTargetLayouts);

  vk::RenderPass getRenderPass() const { return mRenderPass.get(); }
  vk::Pipeline getPipeline() const { return mPipeline.get(); }

private:
  void reflectSPV() override;
  void validate() const;
};


} // namespace shader
} // namespace svulkan2
