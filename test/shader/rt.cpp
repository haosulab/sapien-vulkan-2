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

  RayTracingShaderPackInstance pack({.shaderDir = "../shader/rt",
                                     .maxMeshes = 1000,
                                     .maxMaterials = 1000,
                                     .maxTextures = 1000,
                                     .maxShapes = 1000});
  pack.getPipeline();
  pack.getShaderBindingTable();
}
