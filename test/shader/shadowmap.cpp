#include "svulkan2/shader/shader.h"
#include <gtest/gtest.h>
#include <filesystem>

using namespace svulkan2;
using namespace svulkan2::shader;

TEST(Shadow, Minimal) {
  ShadowPassParser shadow;
  GLSLCompiler::InitializeProcess();
  shadow.loadGLSLFiles("../test/assets/shader/shadowmap_minimal.vert",
                        "../test/assets/shader/shadowmap_minimal.frag");
  GLSLCompiler::FinalizeProcess();

  // vertex
  auto vertexLayout = shadow.getVertexInputLayout();
  ASSERT_EQ(vertexLayout->elements.size(), 1);
  ASSERT_TRUE(CONTAINS(vertexLayout->elements, "position"));
  ASSERT_EQ(vertexLayout->elements["position"].name, "position");
  ASSERT_EQ(vertexLayout->elements["position"].location, 0);
  ASSERT_EQ(vertexLayout->elements["position"].dtype, eFLOAT3);
  ASSERT_EQ(vertexLayout->getSize(), 12);

  // camera
  auto lightSpaceLayout = shadow.getLightSpaceBufferLayout();
  ASSERT_TRUE(CONTAINS(lightSpaceLayout->elements, "lightViewMatrix"));
  ASSERT_TRUE(CONTAINS(lightSpaceLayout->elements, "lightProjectionMatrix"));

  ASSERT_EQ(lightSpaceLayout->elements["lightViewMatrix"].dtype, eFLOAT44);
  ASSERT_EQ(lightSpaceLayout->elements["lightViewMatrix"].size, sizeof(float) * 16);
  ASSERT_EQ(lightSpaceLayout->elements["lightViewMatrix"].arrayDim, 0);
  ASSERT_EQ(lightSpaceLayout->elements["lightViewMatrix"].member, nullptr);
  ASSERT_EQ(lightSpaceLayout->elements["lightViewMatrix"].offset, 0);
  ASSERT_EQ(lightSpaceLayout->elements["lightProjectionMatrix"].offset, 64);
  ASSERT_EQ(lightSpaceLayout->size, 128);

  // object
  auto objectLayout = shadow.getObjectBufferLayout();
  ASSERT_TRUE(CONTAINS(objectLayout->elements, "modelMatrix"));
  ASSERT_TRUE(CONTAINS(objectLayout->elements, "segmentation"));

  ASSERT_EQ(objectLayout->elements["modelMatrix"].dtype, eFLOAT44);
  ASSERT_EQ(objectLayout->elements["modelMatrix"].size, sizeof(float) * 16);
  ASSERT_EQ(objectLayout->elements["modelMatrix"].arrayDim, 0);
  ASSERT_EQ(objectLayout->elements["modelMatrix"].member, nullptr);
  ASSERT_EQ(objectLayout->elements["modelMatrix"].offset, 0);

  ASSERT_EQ(objectLayout->elements["segmentation"].dtype, eUINT4);
  ASSERT_EQ(objectLayout->elements["segmentation"].size, sizeof(uint32_t) * 4);
  ASSERT_EQ(objectLayout->elements["segmentation"].arrayDim, 0);
  ASSERT_EQ(objectLayout->elements["segmentation"].member, nullptr);
  ASSERT_EQ(objectLayout->elements["segmentation"].offset, 64);
  ASSERT_EQ(objectLayout->size, 80);
}
