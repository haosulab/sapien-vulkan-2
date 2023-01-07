#pragma once
#include "svulkan2/core/context.h"
#include "svulkan2/core/descriptor_pool.h"
#include "svulkan2/resource/cubemap.h"
#include "svulkan2/resource/render_target.h"
#include "svulkan2/scene/scene.h"
#include "svulkan2/shader/rt.h"

namespace svulkan2 {
namespace core {
class Context;
class CommandPool;
} // namespace core

namespace renderer {

class RTRenderer {

public:
  RTRenderer(std::string const &shaderDir);

  void render(scene::Camera &camera,
              std::vector<vk::Semaphore> const &waitSemaphores,
              std::vector<vk::PipelineStageFlags> const &waitStages,
              std::vector<vk::Semaphore> const &signalSemaphores,
              vk::Fence fence);

private:
  std::shared_ptr<core::Context> mContext;
  std::shared_ptr<shader::RayTracingShaderPack> mShaderPack;
};

} // namespace renderer
} // namespace svulkan2
