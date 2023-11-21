#include "svulkan2/shader/rt.h"
#include "svulkan2/common/log.h"
#include "svulkan2/core/context.h"
#include "svulkan2/scene/scene.h"
#include "svulkan2/renderer/renderer.h"
#include <gtest/gtest.h>

using namespace svulkan2;
using namespace svulkan2::shader;

TEST(RTCore, Create) {
  log::getLogger()->set_level(spdlog::level::debug);
  auto context = core::Context::Create();
  auto manager = context->createResourceManager();

  auto config = std::make_shared<RendererConfig>();
  config->shaderDir = "../shader/point";
  config->colorFormat4 = vk::Format::eR32G32B32A32Sfloat;
  renderer::Renderer renderer(config);

  svulkan2::scene::Scene scene;
  auto model =
      resource::SVModel::FromFile("../test/assets/scene/dragon/dragon.obj");
  scene.addObject(model);

  model->buildBLAS();

  // RayTracingParser parser;
  // parser
  //     .loadGLSLFilesAsync(
  //         "../shader/rt/camera.rgen",
  //         {"../shader/rt/camera.rmiss", "../shader/rt/shadow.rmiss"},
  //         {"../shader/rt/camera.rahit"}, {"../shader/rt/camera.rchit"})
  //     .get();

  // log::info(parser.summary());
}
