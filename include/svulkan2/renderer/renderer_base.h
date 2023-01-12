#pragma once
#include "svulkan2/core/context.h"
#include "svulkan2/scene/scene.h"
#include "svulkan2/shader/rt.h"

namespace svulkan2 {
namespace renderer {

class RendererBase {
public:
  virtual void resize(int width, int height) = 0;
  virtual void setScene(scene::Scene &scene) = 0;

  virtual void render(scene::Camera &camera,
                      std::vector<vk::Semaphore> const &waitSemaphores,
                      std::vector<vk::PipelineStageFlags> const &waitStages,
                      std::vector<vk::Semaphore> const &signalSemaphores,
                      vk::Fence fence) = 0;

  virtual void render(
      scene::Camera &camera,
      vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &waitSemaphores,
      vk::ArrayProxyNoTemporaries<vk::PipelineStageFlags const> const
          &waitStageMasks,
      vk::ArrayProxyNoTemporaries<uint64_t const> const &waitValues,
      vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &signalSemaphores,
      vk::ArrayProxyNoTemporaries<uint64_t const> const &signalValues) = 0;

  virtual void display(std::string const &imageName, vk::Image backBuffer,
                       vk::Format format, uint32_t width, uint32_t height,
                       std::vector<vk::Semaphore> const &waitSemaphores,
                       std::vector<vk::PipelineStageFlags> const &waitStages,
                       std::vector<vk::Semaphore> const &signalSemaphores,
                       vk::Fence fence) = 0;

  virtual void
  setCustomTexture(std::string const &name,
                   std::shared_ptr<resource::SVTexture> texture) = 0;
  virtual void
  setCustomCubemap(std::string const &name,
                   std::shared_ptr<resource::SVCubemap> cubemap) = 0;

  virtual void setCustomProperty(std::string const &name, int p) {}
  virtual void setCustomProperty(std::string const &name, float p) {}
  virtual void setCustomProperty(std::string const &name, glm::vec3 p) {}
  virtual void setCustomProperty(std::string const &name, glm::vec4 p) {}

  virtual std::vector<std::string> getRenderTargetNames() const = 0;

  virtual ~RendererBase() = default;
};

} // namespace renderer
} // namespace svulkan2
