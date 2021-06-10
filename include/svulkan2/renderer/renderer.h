#pragma once
#include "svulkan2/core/context.h"
#include "svulkan2/core/descriptor_pool.h"
#include "svulkan2/resource/cubemap.h"
#include "svulkan2/resource/render_target.h"
#include "svulkan2/scene/scene.h"
#include "svulkan2/shader/shader_manager.h"
#include <map>

#ifdef CUDA_INTEROP
#include "svulkan2/core/cuda_buffer.h"
#endif

namespace svulkan2 {
namespace renderer {

class Renderer {
  std::shared_ptr<core::Context> mContext;
  std::shared_ptr<RendererConfig> mConfig;

  vk::UniqueDescriptorPool mDescriptorPool;
  std::unique_ptr<core::DynamicDescriptorPool> mObjectPool;

  std::unique_ptr<shader::ShaderManager> mShaderManager;
  std::unordered_map<std::string, std::shared_ptr<resource::SVRenderTarget>>
      mRenderTargets;

  // shadow targets ================================
  uint32_t mNumPointLightShadows{};
  uint32_t mNumDirectionalLightShadows{};
  uint32_t mNumSpotLightShadows{};
  uint32_t mNumCustomShadows{};
  std::shared_ptr<resource::SVRenderTarget> mDirectionalShadowReadTarget;
  std::vector<std::shared_ptr<resource::SVRenderTarget>>
      mDirectionalShadowWriteTargets;

  std::shared_ptr<resource::SVRenderTarget> mPointShadowReadTarget;
  std::vector<std::shared_ptr<resource::SVRenderTarget>>
      mPointShadowWriteTargets;

  std::shared_ptr<resource::SVRenderTarget> mSpotShadowReadTarget;
  std::vector<std::shared_ptr<resource::SVRenderTarget>>
      mSpotShadowWriteTargets;

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
  std::map<std::string, std::shared_ptr<resource::SVCubemap>> mCustomCubemaps;

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

  bool mRequiresRecord{false};
  uint64_t mLastVersion{0};
  scene::Scene *mScene{nullptr};

#ifdef CUDA_INTEROP
  std::map<std::string, std::shared_ptr<core::CudaBuffer>> mCudaBuffers;
#endif
public:
  Renderer(std::shared_ptr<core::Context> context,
           std::shared_ptr<RendererConfig> config);

  void setSpecializationConstantInt(std::string const &name, int value);
  void setSpecializationConstantFloat(std::string const &name, float value);

  void resize(int width, int height);

  void render(scene::Camera &camera,
              std::vector<vk::Semaphore> const &waitSemaphores,
              std::vector<vk::PipelineStageFlags> const &waitStages,
              std::vector<vk::Semaphore> const &signalSemaphores,
              vk::Fence fence);

  void display(std::string const &renderTargetName, vk::Image backBuffer,
               vk::Format format, uint32_t width, uint32_t height,
               std::vector<vk::Semaphore> const &waitSemaphores,
               std::vector<vk::PipelineStageFlags> const &waitStages,
               std::vector<vk::Semaphore> const &signalSemaphores,
               vk::Fence fence);

  void setScene(scene::Scene &scene) {
    mScene = &scene;
    mRequiresRecord = false;
  }

  std::vector<std::string> getDisplayTargetNames() const;
  std::vector<std::string> getRenderTargetNames() const;

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

  template <typename T>
  std::tuple<std::vector<T>, std::array<uint32_t, 3>>
  downloadRegion(std::string const &targetName, vk::Offset2D offset,
                 vk::Extent2D extent) {
    auto target = mRenderTargets.at(targetName);
    uint32_t width = target->getWidth();
    uint32_t height = target->getHeight();
    if (offset.x < 0 || offset.y < 0 || offset.x + extent.width >= width ||
        offset.y + extent.height >= height) {
      throw std::runtime_error(
          "failed to download region: offset or extent is out of bound");
    }
    std::vector<T> data = target->getImage().download<T>(
        vk::Offset3D{offset.x, offset.y, 0},
        vk::Extent3D{extent.width, extent.height, 1});
    uint32_t channels = data.size() / (extent.width * extent.height);
    if (extent.width * extent.height * channels != data.size()) {
      throw std::runtime_error(
          "failed to download region: internal format error");
    }
    return {data,
            std::array<uint32_t, 3>{extent.height, extent.width, channels}};
  }

  std::shared_ptr<resource::SVRenderTarget>
  getRenderTarget(std::string const &name) const;

  void setCustomTexture(std::string const &name,
                        std::shared_ptr<resource::SVTexture> texture);
  void setCustomCubemap(std::string const &name,
                        std::shared_ptr<resource::SVCubemap> cubemap);

  Renderer(Renderer const &other) = delete;
  Renderer &operator=(Renderer const &other) = delete;
  Renderer(Renderer &&other) = default;
  Renderer &operator=(Renderer &&other) = default;

#ifdef CUDA_INTEROP
  std::tuple<std::shared_ptr<core::CudaBuffer>, std::array<uint32_t, 2>,
             vk::Format>
  transferToCuda(std::string const &targetName);
#endif

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

  std::unordered_set<std::shared_ptr<resource::SVModel>> mModelCache;
  std::unordered_set<std::shared_ptr<resource::SVLineSet>> mLineSetCache;

  vk::UniqueCommandBuffer mShadowCommandBuffer{};
  vk::UniqueCommandBuffer mRenderCommandBuffer{};
  vk::UniqueCommandBuffer mDisplayCommandBuffer{};

  void prepareObjects(scene::Scene &scene);
  void recordShadows(scene::Scene &scene);
  void recordRenderPasses(scene::Scene &scene);
};

} // namespace renderer
} // namespace svulkan2
