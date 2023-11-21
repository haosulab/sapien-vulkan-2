#include "svulkan2/shader/shader.h"
#include <gtest/gtest.h>

using namespace svulkan2;
using namespace svulkan2::shader;

TEST(ShaderManager, Load) {
  auto config = std::make_shared<RendererConfig>();
  config->shaderDir = "../test/assets/shader/full";
  config->colorFormat = vk::Format::eR8G8B8A8Unorm;
  config->depthFormat = vk::Format::eD32Sfloat;
  ShaderManager manager(config);
}

TEST(ShaderManager, Parse) {
  auto config = std::make_shared<RendererConfig>();
  config->shaderDir = "../test/assets/shader/full";
  config->colorFormat = vk::Format::eR8G8B8A8Unorm;
  config->depthFormat = vk::Format::eD32Sfloat;
  ShaderManager manager(config);
  auto formats = manager.getRenderTargetFormats();

  ASSERT_TRUE(CONTAINS(formats, "Albedo"));
  ASSERT_TRUE(CONTAINS(formats, "Position"));
  ASSERT_TRUE(CONTAINS(formats, "Specular"));
  ASSERT_TRUE(CONTAINS(formats, "Normal"));
  ASSERT_TRUE(CONTAINS(formats, "Segmentation"));
  ASSERT_TRUE(CONTAINS(formats, "Custom"));
  ASSERT_TRUE(CONTAINS(formats, "Depth"));
  ASSERT_TRUE(CONTAINS(formats, "Lighting"));
  ASSERT_TRUE(CONTAINS(formats, "Color"));
  ASSERT_EQ(formats.size(), 9);

  ASSERT_EQ(formats["Albedo"], config->colorFormat);
  ASSERT_EQ(formats["Position"], config->colorFormat);
  ASSERT_EQ(formats["Specular"], config->colorFormat);
  ASSERT_EQ(formats["Normal"], config->colorFormat);
  ASSERT_EQ(formats["Segmentation"], vk::Format::eR32G32B32A32Uint);
  ASSERT_EQ(formats["Custom"], config->colorFormat);
  ASSERT_EQ(formats["Depth"], config->depthFormat);
  ASSERT_EQ(formats["Lighting"], config->colorFormat);
  ASSERT_EQ(formats["Color"], config->colorFormat);
}
