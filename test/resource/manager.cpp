#include "svulkan2/resource/manager.h"
#include "svulkan2/core/context.h"
#include <gtest/gtest.h>

using namespace svulkan2;
using namespace svulkan2::resource;

TEST(ResourceManager, Creation) { SVResourceManager manager; }

TEST(ResourceManager, Image) {
  core::Context context;
  auto &manager = context.getResourceManager();
  manager.setMaterialPipelineType(ShaderConfig::MaterialPipeline::eMETALLIC);
  auto layout = std::make_shared<InputDataLayout>();
  layout->elements["position"] = {
      .name = "position", .location = 0, .dtype = eFLOAT3};
  layout->elements["normal"] = {
      .name = "normal", .location = 1, .dtype = eFLOAT3};
  manager.setVertexLayout(layout);

  auto img = manager.CreateImageFromFile("../test/assets/image/4x2.png", 1);
  ASSERT_FALSE(img->isLoaded());
  img->load();
  ASSERT_TRUE(img->isLoaded());

  // these load should not do anything
  img->load();
  img->load();

  ASSERT_FALSE(img->isOnDevice());
  img->uploadToDevice(context);
  ASSERT_TRUE(img->isOnDevice());

  // these upload should not do anything
  img->uploadToDevice(context);
  img->uploadToDevice(context);

  // creating the same image should result in the same resource
  auto img2 = manager.CreateImageFromFile("../test/../test/assets/image/4x2.png",
                                          1);
  ASSERT_TRUE(img2->isLoaded());
  ASSERT_TRUE(img2->isOnDevice());
}

TEST(ResourceManager, Texture) {
  core::Context context;
  auto &manager = context.getResourceManager();
  manager.setMaterialPipelineType(ShaderConfig::MaterialPipeline::eMETALLIC);
  auto layout = std::make_shared<InputDataLayout>();
  layout->elements["position"] = {
      .name = "position", .location = 0, .dtype = eFLOAT3};
  layout->elements["normal"] = {
      .name = "normal", .location = 1, .dtype = eFLOAT3};
  manager.setVertexLayout(layout);

  auto tex1 = manager.CreateTextureFromFile("../test/assets/image/4x2.png", 1);
  ASSERT_FALSE(tex1->isLoaded());
  tex1->load();
  ASSERT_TRUE(tex1->isLoaded());
  tex1->load();
  tex1->load();

  ASSERT_FALSE(tex1->isOnDevice());
  tex1->uploadToDevice(context);
  ASSERT_TRUE(tex1->isOnDevice());
  tex1->uploadToDevice(context);
  tex1->uploadToDevice(context);

  // same texture
  auto tex2 = manager.CreateTextureFromFile("../test/assets/image/4x2.png", 1);
  ASSERT_TRUE(tex2->isLoaded());
  ASSERT_TRUE(tex2->isOnDevice());

  // different texture same image
  auto tex3 = manager.CreateTextureFromFile("../test/assets/image/4x2.png", 1, vk::Filter::eNearest);
  ASSERT_FALSE(tex3->isLoaded());
  tex3->load();
  ASSERT_TRUE(tex3->isLoaded());
  tex3->load();
  tex3->load();

  ASSERT_FALSE(tex3->isOnDevice());
  tex3->uploadToDevice(context);
  ASSERT_TRUE(tex3->isOnDevice());
  tex3->uploadToDevice(context);
  tex3->uploadToDevice(context);

  ASSERT_EQ(manager.getImages().size(), 1);
}
