#pragma once
#include "svulkan2/common/config.h"
#include "svulkan2/core/context.h"
#include "svulkan2/shader/composite.h"
#include "svulkan2/shader/deferred.h"
#include "svulkan2/shader/gbuffer.h"
#include <map>
#include <memory>
#include <string>

namespace svulkan2 {

namespace shader {
enum class TextureOperation { eTextureNoOp, eTextureRead, eTextureWrite };

class ShaderManager {
  uint32_t mNumPasses{};
  std::shared_ptr<RendererConfig> mRenderConfig;
  std::shared_ptr<ShaderConfig> mShaderConfig;

  std::shared_ptr<GbufferPassParser> mGbufferPass;
  std::shared_ptr<DeferredPassParser> mDeferredPass;
  std::vector<std::shared_ptr<CompositePassParser>> mCompositePasses;
  std::map<std::weak_ptr<BaseParser>, unsigned int, std::owner_less<>>
      mPassIndex;
  std::unordered_map<std::string, std::vector<TextureOperation>>
      mTextureOperationTable;
  std::unordered_map<std::string, vk::Format> mRenderTargetFormats;

  vk::UniqueDescriptorSetLayout mSceneLayout;
  vk::UniqueDescriptorSetLayout mCameraLayout;
  vk::UniqueDescriptorSetLayout mObjectLayout;
  vk::UniqueDescriptorSetLayout mDeferredLayout;
  std::vector<vk::UniqueDescriptorSetLayout> mCompositeLayouts;

public:
  ShaderManager(std::shared_ptr<RendererConfig> config = nullptr);

  std::shared_ptr<RendererConfig> getConfig() const { return mRenderConfig; }
  std::shared_ptr<GbufferPassParser> getGbufferPass() const {
    return mGbufferPass;
  }
  std::shared_ptr<DeferredPassParser> getDeferredPass() const {
    return mDeferredPass;
  }
  std::vector<std::shared_ptr<CompositePassParser>> getCompositePasses() const {
    return mCompositePasses;
  }

  inline vk::DescriptorSetLayout getSceneDescriptorSetLayout() const {
    return mSceneLayout.get();
  }
  inline vk::DescriptorSetLayout getCameraDescriptorSetLayout() const {
    return mCameraLayout.get();
  }
  inline vk::DescriptorSetLayout getObjectDescriptorSetLayout() const {
    return mObjectLayout.get();
  }
  inline vk::DescriptorSetLayout getDeferredDescriptorSetLayout() const {
    return mDeferredLayout.get();
  }
  std::vector<vk::DescriptorSetLayout> getCompositeDescriptorSetLayout() const;

  // std::vector<vk::Pipeline> getPipelines() const { return mPipelines; }
  // std::vector<vk::RenderPass> getRenderPasses() const { return mRenderPasses;
  // }
  // std::vector<vk::PipelineLayout>
  // getPipelinesLayouts(); // call only after createPipelines.

  void createPipelines(core::Context &context, int numDirectionalLights = -1,
                       int numPointLights = -1);
  std::vector<std::shared_ptr<BaseParser>> getAllPasses() const;

  inline std::unordered_map<std::string, vk::Format>
  getRenderTargetFormats() const {
    return mRenderTargetFormats;
  };

  inline std::shared_ptr<ShaderConfig> getShaderConfig() const {
    return mShaderConfig;
  }

private:
  void processShadersInFolder(std::string const &folder);
  void createDescriptorSetLayouts(vk::Device device);
  void populateShaderConfig();
  void prepareRenderTargetFormats();
  void prepareRenderTargetOperationTable();
  TextureOperation getNextOperation(std::string texName,
                                    std::shared_ptr<BaseParser> pass);
  TextureOperation getPrevOperation(std::string texName,
                                    std::shared_ptr<BaseParser> pass);
  // std::unordered_map<std::string, std::pair<vk::ImageLayout,
  // vk::ImageLayout>> getRenderTargetLayouts(std::shared_ptr<BaseParser> pass,
  //                   std::shared_ptr<OutputDataLayout> outputLayout);
  std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>>
  getColorRenderTargetLayoutsForPass(std::shared_ptr<BaseParser> pass);
};

} // namespace shader
} // namespace svulkan2
