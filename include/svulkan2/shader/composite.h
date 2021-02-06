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
  inline std::shared_ptr<CombinedSamplerLayout> getCombinedSamplerLayout() const {
    return mCombinedSamplerLayout;
  }
  inline std::shared_ptr<OutputDataLayout> getTextureOutputLayout() const {
    return mTextureOutputLayout;
  }

  vk::RenderPass getRenderPass() const { return mRenderPass.get(); }
  vk::Pipeline getPipeline() const { return mPipeline.get(); }

  vk::PipelineLayout createPipelineLayout(vk::Device device) override;
  vk::RenderPass createRenderPass(vk::Device device, vk::Format colorFormat,
      vk::Format depthFormat, std::unordered_map<std::string, std::pair<vk::ImageLayout, vk::ImageLayout>> layouts);

  vk::Pipeline
      createGraphicsPipeline(vk::Device device,
          vk::Format colorFormat, vk::Format depthFormat, std::unordered_map<std::string, std::pair<vk::ImageLayout, vk::ImageLayout>> renderTargetLayouts);

private:
  void reflectSPV() override;
  void validate() const;
};

} // namespace shader
} // namespace svulkan2
