#pragma once
#include "svulkan2/common/config.h"
#include "svulkan2/common/layout.h"
#include "svulkan2/shader/shader_pack.h"
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

namespace svulkan2 {
namespace core {
class Context;
}
namespace shader {

struct ShaderPackInstanceDesc {
  std::shared_ptr<RendererConfig> config;
  std::map<std::string, SpecializationConstantValue> specializationConstants;

  bool operator==(ShaderPackInstanceDesc const &other) const {
    return *config == *other.config && specializationConstants == other.specializationConstants;
  }
};

enum class RenderTargetOperation { eNoOp, eRead, eColorWrite, eDepthWrite };
using optable_t = std::unordered_map<std::string, std::vector<RenderTargetOperation>>;

class ShaderPackInstance {
public:
  ShaderPackInstance(ShaderPackInstanceDesc const &desc);

  std::future<void> loadAsync();

  inline ShaderPackInstanceDesc const &getDesc() const { return mDesc; }

  struct PipelineResources {
    vk::UniquePipelineLayout layout{};
    vk::UniqueRenderPass renderPass{};
    vk::UniquePipeline pipeline{};
  };

  inline std::shared_ptr<ShaderPack> getShaderPack() const { return mShaderPack; }

  inline std::vector<PipelineResources> const &getNonShadowPassResources() const {
    return mNonShadowPassResources;
  }

  inline PipelineResources const &getShadowPassResources() const { return mShadowPassResources; }

  inline PipelineResources const &getPointShadowPassResources() const {
    return mPointShadowPassResources;
  }

  inline std::unordered_map<std::string, vk::Format> getRenderTargetFormats() const {
    return mRenderTargetFormats;
  }

  inline std::unordered_map<std::string, vk::ImageLayout> getRenderTargetFinalLayouts() const {
    return mRenderTargetFinalLayouts;
  }

  inline vk::DescriptorSetLayout getSceneDescriptorSetLayout() const {
    return mSceneSetLayout.get();
  }
  inline vk::DescriptorSetLayout getObjectDescriptorSetLayout() const {
    return mObjectSetLayout.get();
  }
  inline vk::DescriptorSetLayout getCameraDescriptorSetLayout() const {
    return mCameraSetLayout.get();
  }
  inline vk::DescriptorSetLayout getLightDescriptorSetLayout() const {
    return mLightSetLayout.get();
  }

  inline std::vector<vk::DescriptorSetLayout> getInputTextureLayouts() const {
    std::vector<vk::DescriptorSetLayout> result;
    for (auto &layout : mTextureSetLayouts) {
      result.push_back(layout.get());
    }
    return result;
  }

  std::optional<std::string> getDepthRenderTargetName(BaseParser const &pass) const;

private:
  ShaderPackInstanceDesc mDesc;
  std::shared_ptr<ShaderPack> mShaderPack;
  std::shared_ptr<core::Context> mContext;

  vk::UniqueDescriptorSetLayout mSceneSetLayout;
  vk::UniqueDescriptorSetLayout mObjectSetLayout;
  vk::UniqueDescriptorSetLayout mCameraSetLayout;
  vk::UniqueDescriptorSetLayout mLightSetLayout;
  std::vector<vk::UniqueDescriptorSetLayout> mTextureSetLayouts;

  optable_t mTextureOperationTable;
  optable_t generateTextureOperationTable() const;

  std::vector<PipelineResources> mNonShadowPassResources;
  PipelineResources mShadowPassResources;
  PipelineResources mPointShadowPassResources;

  std::unordered_map<std::string, vk::Format> mRenderTargetFormats;
  std::unordered_map<std::string, vk::ImageLayout> mRenderTargetFinalLayouts;

  std::mutex mLoadingMutex;
  bool mLoaded{};
};

} // namespace shader
} // namespace svulkan2
