#pragma once
#include "svulkan2/common/config.h"
#include "svulkan2/core/context.h"
#include "svulkan2/shader/deferred.h"
#include "svulkan2/shader/gbuffer.h"
#include <map>
#include <memory>
#include <string>

namespace svulkan2 {

namespace shader {
enum class RenderTargetOperation { eNoOp, eRead, eColorWrite, eDepthWrite };

class ShaderManager {
  std::shared_ptr<RendererConfig> mRenderConfig;
  std::shared_ptr<ShaderConfig> mShaderConfig;

  std::vector<std::shared_ptr<BaseParser>> mAllPasses;
  std::map<std::weak_ptr<BaseParser>, unsigned int, std::owner_less<>>
      mPassIndex;

  std::unordered_map<std::string, std::vector<RenderTargetOperation>>
      mTextureOperationTable;
  std::unordered_map<std::string, vk::Format> mRenderTargetFormats;

  vk::UniqueDescriptorSetLayout mSceneLayout;
  vk::UniqueDescriptorSetLayout mCameraLayout;
  vk::UniqueDescriptorSetLayout mObjectLayout;

  std::vector<vk::UniqueDescriptorSetLayout> mInputTextureLayouts;

public:
  ShaderManager(std::shared_ptr<RendererConfig> config = nullptr);

  std::shared_ptr<RendererConfig> getConfig() const { return mRenderConfig; }

  inline vk::DescriptorSetLayout getSceneDescriptorSetLayout() const {
    return mSceneLayout.get();
  }
  inline vk::DescriptorSetLayout getCameraDescriptorSetLayout() const {
    return mCameraLayout.get();
  }
  inline vk::DescriptorSetLayout getObjectDescriptorSetLayout() const {
    return mObjectLayout.get();
  }

  std::vector<vk::DescriptorSetLayout> getInputTextureLayouts() const;

  void createPipelines(core::Context &context,
                       std::map<std::string, SpecializationConstantValue> const
                           &specializationConstantInfo);
  std::vector<std::shared_ptr<BaseParser>> getAllPasses() const;

  inline std::unordered_map<std::string, vk::Format>
  getRenderTargetFormats() const {
    return mRenderTargetFormats;
  };

  std::unordered_map<std::string, vk::ImageLayout>
  getRenderTargetFinalLayouts() const;

  inline std::shared_ptr<ShaderConfig> getShaderConfig() const {
    return mShaderConfig;
  }

private:
  void processShadersInFolder(std::string const &folder);
  void createDescriptorSetLayouts(vk::Device device);
  void populateShaderConfig();
  void prepareRenderTargetFormats();
  void prepareRenderTargetOperationTable();
  RenderTargetOperation getNextOperation(std::string texName,
                                         std::shared_ptr<BaseParser> pass);
  RenderTargetOperation getPrevOperation(std::string texName,
                                         std::shared_ptr<BaseParser> pass);
  RenderTargetOperation getLastOperation(std::string texName) const;

  std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>>
  getColorAttachmentLayoutsForPass(std::shared_ptr<BaseParser> pass);
  std::pair<vk::ImageLayout, vk::ImageLayout>
  getDepthAttachmentLayoutsForPass(std::shared_ptr<BaseParser> pass);
};

} // namespace shader
} // namespace svulkan2
