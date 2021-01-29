#pragma once
#include "svulkan2/core/context.h"
#include "svulkan2/resource/camera.h"
#include "svulkan2/resource/render_target.h"
#include "svulkan2/scene/scene.h"
#include "svulkan2/shader/manager.h"
#include <map>

namespace svulkan2 {
namespace renderer {

class Renderer {
  core::Context *mContext;
  std::unique_ptr<shader::ShaderManager> mShaderManager;
  std::map<std::string, std::shared_ptr<resource::SVRenderTarget>>
      mRenderTargets;

  int mWidth{};
  int mHeight{};

public:
  Renderer(RendererConfig const &config);

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
};

} // namespace renderer
} // namespace svulkan2
