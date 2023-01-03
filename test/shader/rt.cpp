#include "svulkan2/shader/rt.h"
#include "svulkan2/common/log.h"
#include "svulkan2/core/context.h"
#include <gtest/gtest.h>

using namespace svulkan2;
using namespace svulkan2::shader;

TEST(RTShader, Create) {
  log::getLogger()->set_level(spdlog::level::debug);
  auto context = core::Context::Create();
  auto manager = context->createResourceManager();

  RayTracingParser parser;
  parser
      .loadGLSLFilesAsync(
          "../shader/rt/camera.rgen",
          {"../shader/rt/camera.rmiss", "../shader/rt/shadow.rmiss"},
          {"../shader/rt/camera.rahit"}, {"../shader/rt/camera.rchit"})
      .get();

  log::info(parser.summary());
}
