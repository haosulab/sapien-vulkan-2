#include "svulkan2/common/fs.h"
#include "svulkan2/common/log.h"
#include "svulkan2/core/context.h"
#include "svulkan2/renderer/renderer.h"
#include "svulkan2/scene/scene.h"
#include "svulkan2/shader/shader.h"
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

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
  config->colorFormat = vk::Format::eR8G8B8A8Unorm;
  renderer::Renderer renderer(context, config);

  svulkan2::scene::Scene scene;

  scene.setSVScene(std::make_shared<resource::SVScene>());
  scene.getSVScene()->addPointLight({{1, 0, 0, 1}, {1, 1, 1, 1}});
  scene.getSVScene()->addDirectionalLight({{0, 0, -1, 1}, {1, 1, 1, 1}});

  auto node = scene.addNode();
  auto model = context.getResourceManager().CreateModelFromFile(
      "../test/assets/scene/sponza/sponza.obj");
  node.setTransform(scene::Transform{.scale = {0.01, 0.01, 0.01}});
  model->loadAsync().get();
  auto object = std::make_shared<resource::SVObject>(model);
  node.setObject(object);

  auto cameraNode = scene.addNode();
  cameraNode.setTransform(scene::Transform{.position{0, 0, 0.3}});
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

  auto [output, size] = renderer.download<uint8_t>("Albedo");
  std::cout << output.size() << std::endl;
  std::cout << size[0] << " " << size[1] << " " << size[2] << std::endl;

  stbi_write_png("out.png", size[1], size[0], size[2], output.data(),
                 size[1] * size[2]);

  return 0;
}
