#pragma once
#include "svulkan2/core/context.h"
#include "svulkan2/resource/camera.h"
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
  std::map<std::string, std::shared_ptr<resource::SVRenderTarget>>
      mRenderTargets;

  // std::vector<vk::Pipeline> mPipelines;
  // std::vector<vk::RenderPass> mRenderPasses;
  std::vector<vk::UniqueFramebuffer> mFramebuffers;

  int mWidth{};
  int mHeight{};
  bool mRequiresRebuild{true};

  std::unique_ptr<core::Buffer> mSceneBuffer;
  std::unique_ptr<core::Buffer> mCameraBuffer;
  std::vector<std::unique_ptr<core::Buffer>> mObjectBuffers;

  vk::UniqueDescriptorSet mSceneSet;
  vk::UniqueDescriptorSet mCameraSet;
  std::vector<vk::UniqueDescriptorSet> mObjectSet;
  vk::UniqueDescriptorSet mDeferredSet;
  std::vector<vk::UniqueDescriptorSet> mCompositeSets;

  bool mLastNumPointLights = 0;
  bool mLastNumDirectionalLights = 0;
public:
  Renderer(core::Context &context, std::shared_ptr<RendererConfig> config);

  void resize(int width, int height);

  void render(vk::CommandBuffer commandBuffer, scene::Scene &scene,
              resource::SVCamera &camera);

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

  Renderer(Renderer const &other) = delete;
  Renderer &operator=(Renderer const &other) = delete;
  Renderer(Renderer &&other) = default;
  Renderer &operator=(Renderer &&other) = default;

private:
  void prepareRenderTargets(uint32_t width, uint32_t height);
  void preparePipelines(int numDirectionalLights, int numPointLights);
  void prepareFramebuffers(uint32_t width, uint32_t height);

  void prepareSceneBuffer();
  void prepareObjectBuffers(uint32_t numObjects);
  void prepareCameaBuffer();
};

} // namespace renderer
} // namespace svulkan2
