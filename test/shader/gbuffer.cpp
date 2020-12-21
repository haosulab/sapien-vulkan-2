#include "svulkan2/shader/shader.h"
#include <gtest/gtest.h>

using namespace svulkan2;
using namespace svulkan2::shader;

TEST(Gbuffer, Minimal) {
  GbufferPassParser gbuffer;
  gbuffer.loadGLSLFiles("../test/assets/shader/gbuffer_minimal.vert",
                        "../test/assets/shader/gbuffer_minimal.frag");

  // vertex
  auto vertexLayout = gbuffer.getVertexInputLayout();
  ASSERT_EQ(vertexLayout->elements.size(), 1);
  ASSERT_TRUE(CONTAINS(vertexLayout->elements, "position"));
  ASSERT_EQ(vertexLayout->elements["position"].name, "position");
  ASSERT_EQ(vertexLayout->elements["position"].location, 0);
  ASSERT_EQ(vertexLayout->elements["position"].dtype, eFLOAT3);
  ASSERT_EQ(vertexLayout->getSize(), 12);

  // camera
  auto cameraLayout = gbuffer.getCameraBufferLayout();
  ASSERT_TRUE(CONTAINS(cameraLayout->elements, "viewMatrix"));
  ASSERT_TRUE(CONTAINS(cameraLayout->elements, "projectionMatrix"));
  ASSERT_TRUE(CONTAINS(cameraLayout->elements, "viewMatrixInverse"));
  ASSERT_TRUE(CONTAINS(cameraLayout->elements, "projectionMatrixInverse"));

  ASSERT_EQ(cameraLayout->elements["viewMatrix"].dtype, eFLOAT44);
  ASSERT_EQ(cameraLayout->elements["viewMatrix"].size, sizeof(float) * 16);
  ASSERT_EQ(cameraLayout->elements["viewMatrix"].arrayDim, 0);
  ASSERT_EQ(cameraLayout->elements["viewMatrix"].member, nullptr);
  ASSERT_EQ(cameraLayout->elements["viewMatrix"].offset, 0);
  ASSERT_EQ(cameraLayout->elements["projectionMatrix"].offset, 64);
  ASSERT_EQ(cameraLayout->elements["viewMatrixInverse"].offset, 128);
  ASSERT_EQ(cameraLayout->elements["projectionMatrixInverse"].offset, 192);
  ASSERT_EQ(cameraLayout->size, 256);

  // object
  auto objectLayout = gbuffer.getObjectBufferLayout();
  ASSERT_TRUE(CONTAINS(objectLayout->elements, "modelMatrix"));
  ASSERT_TRUE(CONTAINS(objectLayout->elements, "segmentation"));

  ASSERT_EQ(objectLayout->elements["modelMatrix"].dtype, eFLOAT44);
  ASSERT_EQ(objectLayout->elements["modelMatrix"].size, sizeof(float) * 16);
  ASSERT_EQ(objectLayout->elements["modelMatrix"].arrayDim, 0);
  ASSERT_EQ(objectLayout->elements["modelMatrix"].member, nullptr);
  ASSERT_EQ(objectLayout->elements["modelMatrix"].offset, 0);

  ASSERT_EQ(objectLayout->elements["segmentation"].dtype, eUINT4);
  ASSERT_EQ(objectLayout->elements["segmentation"].size, sizeof(uint) * 4);
  ASSERT_EQ(objectLayout->elements["segmentation"].arrayDim, 0);
  ASSERT_EQ(objectLayout->elements["segmentation"].member, nullptr);
  ASSERT_EQ(objectLayout->elements["segmentation"].offset, 64);
  ASSERT_EQ(objectLayout->size, 80);

  // material
  auto materialLayout = gbuffer.getMaterialBufferLayout();
  ASSERT_TRUE(CONTAINS(materialLayout->elements, "baseColor"));
  ASSERT_TRUE(CONTAINS(materialLayout->elements, "fresnel"));
  ASSERT_TRUE(CONTAINS(materialLayout->elements, "roughness"));
  ASSERT_TRUE(CONTAINS(materialLayout->elements, "metallic"));
  ASSERT_TRUE(CONTAINS(materialLayout->elements, "transparency"));
  ASSERT_TRUE(CONTAINS(materialLayout->elements, "textureMask"));

  ASSERT_EQ(materialLayout->elements["baseColor"].dtype, eFLOAT4);
  ASSERT_EQ(materialLayout->elements["baseColor"].size, 16);
  ASSERT_EQ(materialLayout->elements["baseColor"].arrayDim, 0);
  ASSERT_EQ(materialLayout->elements["baseColor"].member, nullptr);
  ASSERT_EQ(materialLayout->elements["baseColor"].offset, 0);

  ASSERT_EQ(materialLayout->elements["fresnel"].dtype, eFLOAT);
  ASSERT_EQ(materialLayout->elements["fresnel"].size, 4);
  ASSERT_EQ(materialLayout->elements["fresnel"].arrayDim, 0);
  ASSERT_EQ(materialLayout->elements["fresnel"].member, nullptr);
  ASSERT_EQ(materialLayout->elements["fresnel"].offset, 16);

  ASSERT_EQ(materialLayout->elements["roughness"].dtype, eFLOAT);
  ASSERT_EQ(materialLayout->elements["roughness"].size, 4);
  ASSERT_EQ(materialLayout->elements["roughness"].arrayDim, 0);
  ASSERT_EQ(materialLayout->elements["roughness"].member, nullptr);
  ASSERT_EQ(materialLayout->elements["roughness"].offset, 20);

  ASSERT_EQ(materialLayout->elements["metallic"].dtype, eFLOAT);
  ASSERT_EQ(materialLayout->elements["metallic"].size, 4);
  ASSERT_EQ(materialLayout->elements["metallic"].arrayDim, 0);
  ASSERT_EQ(materialLayout->elements["metallic"].member, nullptr);
  ASSERT_EQ(materialLayout->elements["metallic"].offset, 24);

  ASSERT_EQ(materialLayout->elements["transparency"].dtype, eFLOAT);
  ASSERT_EQ(materialLayout->elements["transparency"].size, 4);
  ASSERT_EQ(materialLayout->elements["transparency"].arrayDim, 0);
  ASSERT_EQ(materialLayout->elements["transparency"].member, nullptr);
  ASSERT_EQ(materialLayout->elements["transparency"].offset, 28);

  ASSERT_EQ(materialLayout->elements["textureMask"].dtype, eINT);
  ASSERT_EQ(materialLayout->elements["textureMask"].size, 4);
  ASSERT_EQ(materialLayout->elements["textureMask"].arrayDim, 0);
  ASSERT_EQ(materialLayout->elements["textureMask"].member, nullptr);
  ASSERT_EQ(materialLayout->elements["textureMask"].offset, 32);

  ASSERT_EQ(materialLayout->size, 36);

  // texture
  auto samplerLayout = gbuffer.getCombinedSamplerLayout();
  ASSERT_TRUE(CONTAINS(samplerLayout->elements, "colorTexture"));
  ASSERT_TRUE(CONTAINS(samplerLayout->elements, "roughnessTexture"));
  ASSERT_TRUE(CONTAINS(samplerLayout->elements, "normalTexture"));
  ASSERT_TRUE(CONTAINS(samplerLayout->elements, "metallicTexture"));

  ASSERT_EQ(samplerLayout->elements["colorTexture"].binding, 1);
  ASSERT_EQ(samplerLayout->elements["roughnessTexture"].binding, 2);
  ASSERT_EQ(samplerLayout->elements["normalTexture"].binding, 3);
  ASSERT_EQ(samplerLayout->elements["metallicTexture"].binding, 4);

  ASSERT_EQ(samplerLayout->elements["colorTexture"].set, 3);
  ASSERT_EQ(samplerLayout->elements["roughnessTexture"].set, 3);
  ASSERT_EQ(samplerLayout->elements["normalTexture"].set, 3);
  ASSERT_EQ(samplerLayout->elements["metallicTexture"].set, 3);

  // output
  auto outputLayout = gbuffer.getTextureOutputLayout();
  ASSERT_TRUE(CONTAINS(outputLayout->elements, "outAlbedo"));
  ASSERT_TRUE(CONTAINS(outputLayout->elements, "outPosition"));
  ASSERT_TRUE(CONTAINS(outputLayout->elements, "outSpecular"));
  ASSERT_TRUE(CONTAINS(outputLayout->elements, "outNormal"));
  ASSERT_TRUE(CONTAINS(outputLayout->elements, "outSegmentation"));

  ASSERT_EQ(outputLayout->elements["outAlbedo"].location, 0);
  ASSERT_EQ(outputLayout->elements["outPosition"].location, 1);
  ASSERT_EQ(outputLayout->elements["outSpecular"].location, 2);
  ASSERT_EQ(outputLayout->elements["outNormal"].location, 3);
  ASSERT_EQ(outputLayout->elements["outSegmentation"].location, 4);

  ASSERT_EQ(outputLayout->elements["outAlbedo"].dtype, eFLOAT4);
  ASSERT_EQ(outputLayout->elements["outPosition"].dtype, eFLOAT4);
  ASSERT_EQ(outputLayout->elements["outSpecular"].dtype, eFLOAT4);
  ASSERT_EQ(outputLayout->elements["outNormal"].dtype, eFLOAT4);
  ASSERT_EQ(outputLayout->elements["outSegmentation"].dtype, eUINT4);
}
