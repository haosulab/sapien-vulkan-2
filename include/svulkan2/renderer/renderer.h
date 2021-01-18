#pragma once
#include "svulkan2/core/context.h"
#include "svulkan2/resource/camera.h"
#include "svulkan2/scene/scene.h"

namespace svulkan2 {
namespace renderer {

class Renderer {
  core::Context *mContext;

  int mWidth{};
  int mHeight{};

public:
  Renderer();

  void resize(int width, int height);

  void render(vk::CommandBuffer commandBuffer, scene::Scene &scene,
              resource::SVCamera &camera);

  Renderer(Renderer const &other) = delete;
  Renderer &operator=(Renderer const &other) = delete;
  Renderer(Renderer &&other) = default;
  Renderer &operator=(Renderer &&other) = default;

  std::vector<float> download(std::string const &targetName);
};

} // namespace renderer
} // namespace svulkan2
