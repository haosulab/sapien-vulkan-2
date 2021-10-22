#pragma once
#include "svulkan2/common/config.h"
#include "svulkan2/core/context.h"
#include "svulkan2/shader/deferred.h"
#include "svulkan2/shader/gbuffer.h"
#include "svulkan2/shader/line.h"
#include "svulkan2/shader/shadow.h"
#include <map>
#include <memory>
#include <string>

namespace svulkan2 {
namespace core {
class Context;
}

namespace shader {
enum class RenderTargetOperation { eNoOp, eRead, eColorWrite, eDepthWrite };

class ShaderManager {
  std::shared_ptr<core::Context> mContext;
  std::shared_ptr<RendererConfig> mRenderConfig;
  std::shared_ptr<ShaderConfig> mShaderConfig;

  std::shared_ptr<ShadowPassParser> mShadowPass{};
  std::vector<std::shared_ptr<BaseParser>> mAllPasses;
  std::map<std::weak_ptr<BaseParser>, unsigned int, std::owner_less<>>
      mPassIndex;

  std::unordered_map<std::string, std::vector<RenderTargetOperation>>
      mTextureOperationTable;
  std::unordered_map<std::string, vk::Format> mRenderTargetFormats;

  DescriptorSetDescription mCameraSetDesc;
  DescriptorSetDescription mObjectSetDesc;
  DescriptorSetDescription mSceneSetDesc;
  DescriptorSetDescription mLightSetDesc;

  bool mDescriptorSetLayoutsCreated{false};

  vk::UniqueDescriptorSetLayout mSceneLayout;
  vk::UniqueDescriptorSetLayout mCameraLayout;
  vk::UniqueDescriptorSetLayout mObjectLayout;
  vk::UniqueDescriptorSetLayout mLightLayout;

  std::vector<vk::UniqueDescriptorSetLayout> mInputTextureLayouts;

  uint32_t mNumGbufferPasses{};
  bool mShadowEnabled{};
  bool mLineEnabled{};
  bool mPointEnabled{};

public:
  ShaderManager(std::shared_ptr<RendererConfig> config = nullptr);

  std::shared_ptr<RendererConfig> getConfig() const { return mRenderConfig; }

  inline uint32_t getNumGbufferPasses() const { return mNumGbufferPasses; }

  inline vk::DescriptorSetLayout getSceneDescriptorSetLayout() const {
    return mSceneLayout.get();
  }
  inline vk::DescriptorSetLayout getCameraDescriptorSetLayout() const {
    return mCameraLayout.get();
  }
  inline vk::DescriptorSetLayout getObjectDescriptorSetLayout() const {
    return mObjectLayout.get();
  }
  inline vk::DescriptorSetLayout getLightDescriptorSetLayout() const {
    return mLightLayout.get();
  }

  inline DescriptorSetDescription const &getCameraSetDesc() const {
    return mCameraSetDesc;
  }
  inline DescriptorSetDescription const &getObjectSetDesc() const {
    return mObjectSetDesc;
  }
  inline DescriptorSetDescription const &getSceneSetDesc() const {
    return mSceneSetDesc;
  }
  inline DescriptorSetDescription const &getLightSetDesc() const {
    return mLightSetDesc;
  }

  std::vector<vk::DescriptorSetLayout> getInputTextureLayouts() const;

  void createPipelines(std::map<std::string, SpecializationConstantValue> const
                           &specializationConstantInfo);
  std::vector<std::shared_ptr<BaseParser>> getAllPasses() const;
  inline std::shared_ptr<BaseParser> getShadowPass() const {
    return mShadowPass;
  };

  inline std::unordered_map<std::string, vk::Format>
  getRenderTargetFormats() const {
    return mRenderTargetFormats;
  };

  std::unordered_map<std::string, vk::ImageLayout>
  getRenderTargetFinalLayouts() const;

  inline std::shared_ptr<ShaderConfig> getShaderConfig() const {
    return mShaderConfig;
  }

  bool isShadowEnabled() const { return mShadowEnabled; }
  bool isLineEnabled() const { return mLineEnabled; }
  bool isPointEnabled() const { return mPointEnabled; }

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
