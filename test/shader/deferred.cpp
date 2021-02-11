#include "svulkan2/shader/shader.h"
#include <gtest/gtest.h>

using namespace svulkan2;
using namespace svulkan2::shader;

TEST(Deferred, Minimal) {
  DeferredPassParser deferred;
  GLSLCompiler::InitializeProcess();
  deferred.loadGLSLFiles("../test/assets/shader/deferred_minimal.vert",
                         "../test/assets/shader/deferred_minimal.frag");
  GLSLCompiler::FinalizeProcess();

  // specialization Constant
  auto specializationConstantLayout =
      deferred.getSpecializationConstantLayout();
  ASSERT_TRUE(CONTAINS(specializationConstantLayout->elements,
                       "NUM_DIRECTIONAL_LIGHTS"));
  ASSERT_TRUE(
      CONTAINS(specializationConstantLayout->elements, "NUM_POINT_LIGHTS"));

  ASSERT_EQ(
      specializationConstantLayout->elements["NUM_DIRECTIONAL_LIGHTS"].dtype,
      eINT);
  ASSERT_EQ(specializationConstantLayout->elements["NUM_POINT_LIGHTS"].dtype,
            eINT);

  // camera
  auto cameraLayout = deferred.getCameraBufferLayout();
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

  // scene
  auto sceneLayout = deferred.getSceneBufferLayout();
  ASSERT_TRUE(CONTAINS(sceneLayout->elements, "ambientLight"));
  ASSERT_TRUE(CONTAINS(sceneLayout->elements, "directionalLights"));
  ASSERT_TRUE(CONTAINS(sceneLayout->elements, "pointLights"));
  ASSERT_TRUE(
      CONTAINS(sceneLayout->elements["directionalLights"].member->elements,
               "direction"));
  ASSERT_TRUE(CONTAINS(
      sceneLayout->elements["directionalLights"].member->elements, "emission"));
  ASSERT_TRUE(CONTAINS(sceneLayout->elements["pointLights"].member->elements,
                       "position"));
  ASSERT_TRUE(CONTAINS(sceneLayout->elements["pointLights"].member->elements,
                       "emission"));

  ASSERT_EQ(sceneLayout->elements["ambientLight"].dtype, eFLOAT4);
  ASSERT_EQ(sceneLayout->elements["directionalLights"].dtype, eSTRUCT);
  ASSERT_EQ(sceneLayout->elements["directionalLights"].size, 32);
  ASSERT_EQ(sceneLayout->elements["directionalLights"].member->elements.size(),
            2);
  ASSERT_EQ(sceneLayout->elements["directionalLights"]
                .member->elements["direction"]
                .offset,
            0);
  ASSERT_EQ(sceneLayout->elements["directionalLights"]
                .member->elements["direction"]
                .dtype,
            eFLOAT4);
  ASSERT_EQ(sceneLayout->elements["directionalLights"]
                .member->elements["emission"]
                .offset,
            16);
  ASSERT_EQ(sceneLayout->elements["directionalLights"]
                .member->elements["emission"]
                .dtype,
            eFLOAT4);

  ASSERT_EQ(sceneLayout->elements["pointLights"].dtype, eSTRUCT);
  ASSERT_EQ(sceneLayout->elements["pointLights"].size, 32);
  ASSERT_EQ(sceneLayout->elements["pointLights"].member->elements.size(), 2);
  ASSERT_EQ(
      sceneLayout->elements["pointLights"].member->elements["position"].offset,
      0);
  ASSERT_EQ(
      sceneLayout->elements["pointLights"].member->elements["position"].dtype,
      eFLOAT4);
  ASSERT_EQ(
      sceneLayout->elements["pointLights"].member->elements["emission"].offset,
      16);
  ASSERT_EQ(
      sceneLayout->elements["pointLights"].member->elements["emission"].dtype,
      eFLOAT4);

  // texture
  auto samplerLayout = deferred.getCombinedSamplerLayout();
  ASSERT_TRUE(CONTAINS(samplerLayout->elements, "samplerAlbedo"));
  ASSERT_TRUE(CONTAINS(samplerLayout->elements, "samplerPosition"));
  ASSERT_TRUE(CONTAINS(samplerLayout->elements, "samplerSpecular"));
  ASSERT_TRUE(CONTAINS(samplerLayout->elements, "samplerNormal"));

  ASSERT_EQ(samplerLayout->elements["samplerAlbedo"].binding, 0);
  ASSERT_EQ(samplerLayout->elements["samplerPosition"].binding, 1);
  ASSERT_EQ(samplerLayout->elements["samplerSpecular"].binding, 2);
  ASSERT_EQ(samplerLayout->elements["samplerNormal"].binding, 3);

  ASSERT_EQ(samplerLayout->elements["samplerNormal"].set, 2);
  ASSERT_EQ(samplerLayout->elements["samplerPosition"].set, 2);
  ASSERT_EQ(samplerLayout->elements["samplerSpecular"].set, 2);
  ASSERT_EQ(samplerLayout->elements["samplerNormal"].set, 2);

  // output
  auto outputLayout = deferred.getTextureOutputLayout();
  ASSERT_TRUE(CONTAINS(outputLayout->elements, "outLighting2"));

  ASSERT_EQ(outputLayout->elements["outLighting2"].location, 0);
  ASSERT_EQ(outputLayout->elements["outLighting2"].dtype, eFLOAT4);
}
