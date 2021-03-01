#pragma once
#include "svulkan2/core/context.h"
#include "svulkan2/resource/render_target.h"
#include "svulkan2/scene/scene.h"
#include "svulkan2/shader/shader_manager.h"
#include <map>

namespace svulkan2 {
namespace renderer {

class Renderer {
  core::Context *mContext;
  std::shared_ptr<RendererConfig> mConfig;

  vk::UniqueDescriptorPool mDescriptorPool;

  std::unique_ptr<shader::ShaderManager> mShaderManager;
  std::unordered_map<std::string, std::shared_ptr<resource::SVRenderTarget>>
      mRenderTargets;

  // shadow targets ================================
  uint32_t mNumPointLightShadows{};
  uint32_t mNumDirectionalLightShadows{};
  uint32_t mNumCustomShadows{};
  std::shared_ptr<resource::SVRenderTarget> mDirectionalShadowReadTarget;
  std::vector<std::shared_ptr<resource::SVRenderTarget>>
      mDirectionalShadowWriteTargets;

  std::shared_ptr<resource::SVRenderTarget> mPointShadowReadTarget;
  std::vector<std::shared_ptr<resource::SVRenderTarget>>
      mPointShadowWriteTargets;

  std::shared_ptr<resource::SVRenderTarget> mCustomShadowReadTarget;
  std::vector<std::shared_ptr<resource::SVRenderTarget>>
      mCustomShadowWriteTargets;

  std::vector<vk::UniqueDescriptorSet> mLightSets;
  std::vector<std::unique_ptr<core::Buffer>> mLightBuffers;

  std::unique_ptr<core::Buffer> mShadowBuffer;
  // ================================================

  std::unordered_map<std::string, vk::ImageLayout> mRenderTargetFinalLayouts;

  std::vector<vk::UniqueFramebuffer> mFramebuffers;
  std::vector<vk::UniqueFramebuffer> mShadowFramebuffers;

  std::map<std::string, std::shared_ptr<resource::SVTexture>> mCustomTextures;

  int mWidth{};
  int mHeight{};
  bool mRequiresRebuild{true};

  std::unique_ptr<core::Buffer> mSceneBuffer;
  std::unique_ptr<core::Buffer> mCameraBuffer;
  std::vector<std::unique_ptr<core::Buffer>> mObjectBuffers;

  vk::UniqueDescriptorSet mSceneSet;
  vk::UniqueDescriptorSet mCameraSet;
  std::vector<vk::UniqueDescriptorSet> mObjectSet;

  std::vector<vk::UniqueDescriptorSet> mInputTextureSets;

  bool mSpecializationConstantsChanged = true;
  std::map<std::string, shader::SpecializationConstantValue>
      mSpecializationConstants;

public:
  Renderer(core::Context &context, std::shared_ptr<RendererConfig> config);

  void setSpecializationConstantInt(std::string const &name, int value);
  void setSpecializationConstantFloat(std::string const &name, float value);

  void resize(int width, int height);

  void render(vk::CommandBuffer commandBuffer, scene::Scene &scene,
              scene::Camera &camera);

  void display(vk::CommandBuffer commandBuffer,
               std::string const &renderTargetName, vk::Image backBuffer,
               vk::Format format, uint32_t width, uint32_t height);

  template <typename T>
  std::tuple<std::vector<T>, std::array<uint32_t, 3>>
  download(std::string const &targetName) {
    auto target = mRenderTargets.at(targetName);
    uint32_t width = target->getWidth();
    uint32_t height = target->getHeight();
    std::vector<T> data = target->download<T>();
    uint32_t channels = data.size() / (width * height);
    if (width * height * channels != data.size()) {
      throw std::runtime_error(
          "download render target failed: internal format error");
    }
    return {data, std::array<uint32_t, 3>{height, width, channels}};
  }

  void setCustomTexture(std::string const &name,
                        std::shared_ptr<resource::SVTexture> texture);

  Renderer(Renderer const &other) = delete;
  Renderer &operator=(Renderer const &other) = delete;
  Renderer(Renderer &&other) = default;
  Renderer &operator=(Renderer &&other) = default;

private:
  void prepareRenderTargets(uint32_t width, uint32_t height);
  void prepareShadowRenderTargets();
  void preparePipelines();
  void prepareFramebuffers(uint32_t width, uint32_t height);
  void prepareShadowFramebuffers();

  void prepareSceneBuffer();
  void prepareObjectBuffers(uint32_t numObjects);
  void prepareCameaBuffer();
  void prepareLightBuffers();

  void prepareInputTextureDescriptorSets();

  void renderShadows(vk::CommandBuffer commandBuffer, scene::Scene &scene);
};

} // namespace renderer
} // namespace svulkan2
