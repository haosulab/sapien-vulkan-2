#include "svulkan2/renderer/rt_renderer.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace renderer {
RTRenderer::RTRenderer(std::string const &shaderDir) {
  mContext = core::Context::Get();
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  if (!mContext->isRayTracingAvailable()) {
    log::error("The selected GPU does not support ray tracing");
    return;
  }
  mShaderPack = mContext->getResourceManager()->CreateRTShaderPack(shaderDir);
}

void RTRenderer::render(scene::Camera &camera,
                        std::vector<vk::Semaphore> const &waitSemaphores,
                        std::vector<vk::PipelineStageFlags> const &waitStages,
                        std::vector<vk::Semaphore> const &signalSemaphores,
                        vk::Fence fence) {}

} // namespace renderer
} // namespace svulkan2
