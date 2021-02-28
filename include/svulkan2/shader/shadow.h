#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace shader {

class ShadowPassParser : public BaseParser {
  std::shared_ptr<InputDataLayout> mVertexInputLayout;
  std::vector<DescriptorSetDescription> mDescriptorSetDescriptions;

  vk::UniqueRenderPass mRenderPass;
  vk::UniquePipeline mPipeline;

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

  vk::PipelineLayout
  createPipelineLayout(vk::Device device,
                       std::vector<vk::DescriptorSetLayout> layouts);

  vk::RenderPass createRenderPass(
      vk::Device device, vk::Format depthFormat,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout);

  vk::Pipeline createGraphicsPipeline(
      vk::Device device, vk::Format colorFormat, vk::Format depthFormat,
      vk::CullModeFlags cullMode, vk::FrontFace frontFace,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
          &colorTargetLayouts,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
      std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts,
      std::map<std::string, SpecializationConstantValue> const
          &specializationConstantInfo) override;

  inline vk::RenderPass getRenderPass() const override {
    return mRenderPass.get();
  }
  inline vk::Pipeline getPipeline() const override { return mPipeline.get(); }
  std::vector<UniformBindingType> getUniformBindingTypes() const override;

  inline std::vector<DescriptorSetDescription>
  getDescriptorSetDescriptions() const override {
    return mDescriptorSetDescriptions;
  };

private:
  void reflectSPV() override;
  void validate() const;
};

// class ShadowPassParserWrapper : public BaseParser {
//   std::shared_ptr<ShadowPassParser> mShadowPass;

// public:
//   inline std::shared_ptr<InputDataLayout> getVertexInputLayout() const {
//     return mShadowPass->getVertexInputLayout();
//   }

//   inline std::shared_ptr<OutputDataLayout>
//   getTextureOutputLayout() const override {
//     return mShadowPass->getTextureOutputLayout();
//   };

//   inline std::vector<std::string> getColorRenderTargetNames() const override {
//     return mShadowPass->getColorRenderTargetNames();
//   };
//   std::optional<std::string> getDepthRenderTargetName() const override;

//   vk::Pipeline createGraphicsPipeline(
//       vk::Device device, vk::Format colorFormat, vk::Format depthFormat,
//       vk::CullModeFlags cullMode, vk::FrontFace frontFace,
//       std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
//           &colorTargetLayouts,
//       std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
//       std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts,
//       std::map<std::string, SpecializationConstantValue> const
//           &specializationConstantInfo) override {
//     if (auto pipeline = mShadowPass->getPipeline()) {
//       return pipeline;
//     }
//     return mShadowPass->createGraphicsPipeline(
//         device, colorFormat, depthFormat, cullMode, frontFace,
//         colorTargetLayouts, depthLayout, descriptorSetLayouts,
//         specializationConstantInfo);
//   }

//   inline vk::RenderPass getRenderPass() const override {
//     return mShadowPass->getRenderPass();
//   }

//   inline vk::Pipeline getPipeline() const override {
//     return mShadowPass->getPipeline();
//   }

//   inline std::vector<UniformBindingType>
//   getUniformBindingTypes() const override {
//     return mShadowPass->getUniformBindingTypes();
//   }

//   inline std::vector<DescriptorSetDescription>
//   getDescriptorSetDescriptions() const override {
//     return mShadowPass->getDescriptorSetDescriptions();
//   };

//   inline void reflectSPV() override {}
// };

} // namespace shader
} // namespace svulkan2
