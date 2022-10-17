#include "svulkan2/shader/shader_pack.h"
#include "svulkan2/core/context.h"
#include "svulkan2/shader/shader_pack_instance.h"
#include <gtest/gtest.h>

using namespace svulkan2;
using namespace svulkan2::shader;

TEST(ShaderPack, Minimal) {
  auto context = core::Context::Create();
  // log::getLogger()->set_level(spdlog::level::info);
  ShaderPack pack("../shader/ibl");
  pack.getShaderInputLayouts();
  pack.getTextureOperationTable();
}

TEST(ShaderPackInstance, Create) {
  auto context = core::Context::Create();
  auto manager = context->createResourceManager();

  auto pack = manager->CreateShaderPack("../shader/ibl");

  ShaderPackInstanceDesc desc;
  desc.config = std::make_shared<RendererConfig>();
  desc.config->shaderDir = "../shader/ibl";
  desc.specializationConstants = {};

  auto packInstance = manager->CreateShaderPackInstance(desc);

  ASSERT_EQ(packInstance->getShaderPack(), pack);

  auto packInstance2 = manager->CreateShaderPackInstance(desc);
  ASSERT_EQ(packInstance, packInstance2);

  desc.specializationConstants["NUM_POINT_LIGHTS"] = 3;
  auto packInstance3 = manager->CreateShaderPackInstance(desc);

  ASSERT_NE(packInstance, packInstance3);
}

TEST(ShaderPackInstance, Pipeline) {
  auto context = core::Context::Create();
  auto manager = context->createResourceManager();

  ShaderPackInstanceDesc desc;
  desc.config = std::make_shared<RendererConfig>();
  desc.config->shaderDir = "../shader/ibl";
  desc.specializationConstants = {};
  auto packInstance = manager->CreateShaderPackInstance(desc);

  // double load safety
  auto f1 = packInstance->loadAsync();
  auto f2 = packInstance->loadAsync();
  auto f3 = packInstance->loadAsync();
  auto f4 = packInstance->loadAsync();
  f1.wait();
  f2.wait();
  f3.wait();
  f4.wait();

  auto passes = packInstance->getShaderPack()->getNonShadowPasses();
  ASSERT_EQ(packInstance->getNonShadowPassResources().size(), passes.size());

  auto setDesc = packInstance->getShaderPack()
                     ->getShaderInputLayouts()
                     ->sceneSetDescription;
}
