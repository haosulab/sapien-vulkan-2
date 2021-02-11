#include "svulkan2/shader/shader.h"
#include <gtest/gtest.h>

using namespace svulkan2;
using namespace svulkan2::shader;

TEST(Composite, Minimal) {
  CompositePassParser composite;
  GLSLCompiler::InitializeProcess();
  composite.loadGLSLFiles("../test/assets/shader/composite_minimal.vert",
                          "../test/assets/shader/composite_minimal.frag");
  GLSLCompiler::FinalizeProcess();

  auto outputLayout = composite.getTextureOutputLayout();
  ASSERT_TRUE(CONTAINS(outputLayout->elements, "outLighting2"));
  ASSERT_TRUE(CONTAINS(outputLayout->elements, "outSegmentation"));

  ASSERT_EQ(outputLayout->elements["outLighting2"].location, 0);
  ASSERT_EQ(outputLayout->elements["outLighting2"].dtype, eFLOAT4);
  ASSERT_EQ(outputLayout->elements["outSegmentation"].location, 1);
  ASSERT_EQ(outputLayout->elements["outSegmentation"].dtype, eUINT4);

  auto samplerLayout = composite.getCombinedSamplerLayout();
  ASSERT_TRUE(CONTAINS(samplerLayout->elements, "samplerAlbedo"));
  ASSERT_TRUE(CONTAINS(samplerLayout->elements, "samplerLighting"));

  ASSERT_EQ(samplerLayout->elements["samplerAlbedo"].binding, 0);
  ASSERT_EQ(samplerLayout->elements["samplerLighting"].binding, 1);

  ASSERT_EQ(samplerLayout->elements["samplerAlbedo"].set, 0);
  ASSERT_EQ(samplerLayout->elements["samplerLighting"].set, 0);
}
