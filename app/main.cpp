#include "svulkan2/common/fs.h"
#include "svulkan2/common/log.h"
#include "svulkan2/core/context.h"
#include "svulkan2/renderer/renderer.h"
#include "svulkan2/scene/scene.h"
#include "svulkan2/shader/shader.h"

using namespace svulkan2;

int main() {
  svulkan2::log::getLogger()->set_level(spdlog::level::info);

  svulkan2::core::Context context(VK_API_VERSION_1_1, true, 5000, 5000);
  // auto &manager = context.getResourceManager();
  // manager.setMaterialPipelineType(ShaderConfig::MaterialPipeline::eMETALLIC);
  // auto layout = std::make_shared<InputDataLayout>();
  // layout->elements["position"] = {
  //     .name = "position", .location = 0, .dtype = eFLOAT3};
  // layout->elements["normal"] = {
  //     .name = "normal", .location = 1, .dtype = eFLOAT3};
  // manager.setVertexLayout(layout);

  auto config = std::make_shared<RendererConfig>();
  config->shaderDir = "../test/assets/shader/full";
  renderer::Renderer renderer(context, config);

  svulkan2::scene::Scene scene;
  auto node = scene.addNode();

  // auto model = context.getResourceManager().CreateModelFromFile(
  //     "../test/assets/scene/sponza/sponza.obj");
  // model->loadAsync().get();

  scene.setSVScene(std::make_shared<resource::SVScene>());
  scene.getSVScene()->addPointLight({{1, 0, 0, 1}, {1, 1, 1, 1}});
  scene.getSVScene()->addDirectionalLight({{0, 0, -1, 1}, {1, 1, 1, 1}});

  // auto object = std::make_shared<resource::SVObject>(model);
  // node.setObject(object);

  auto cameraNode = scene.addNode();
  auto camera = std::make_shared<resource::SVCamera>();
  camera->setPerspectiveParameters(0.1, 10, 1, 4.f / 3);
  cameraNode.setCamera(camera);

  renderer.resize(800, 600);

  auto commandBuffer = context.createCommandBuffer();
  commandBuffer->begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  renderer.render(commandBuffer.get(), scene, *camera);
  commandBuffer->end();
  context.submitCommandBufferAndWait(commandBuffer.get());

  return 0;
}
