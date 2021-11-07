#include "svulkan2/shader/shader.h"
#include <gtest/gtest.h>

using namespace svulkan2;
using namespace svulkan2::shader;

TEST(Deferred, Minimal) {
  DeferredPassParser deferred;
  GLSLCompiler::InitializeProcess();
  deferred.loadGLSLFiles("../shader/test/deferred.vert",
                         "../shader/test/deferred.frag");
  GLSLCompiler::FinalizeProcess();

  // specialization Constant
  auto specializationConstant = deferred.getSpecializationConstantLayout();

  ASSERT_TRUE(
      specializationConstant->elements.contains("NUM_DIRECTIONAL_LIGHTS"));
  ASSERT_TRUE(specializationConstant->elements.contains("NUM_POINT_LIGHTS"));

  ASSERT_EQ(specializationConstant->elements["NUM_DIRECTIONAL_LIGHTS"].dtype,
            eINT);
  ASSERT_EQ(specializationConstant->elements["NUM_POINT_LIGHTS"].dtype, eINT);

  auto desc = deferred.getDescriptorSetDescriptions();
  DescriptorSetDescription d;
  // scene
  d = desc[0];
  ASSERT_EQ(d.type, UniformBindingType::eScene);
  ASSERT_EQ(d.bindings.at(0).type, vk::DescriptorType::eUniformBuffer);
  ASSERT_EQ(d.bindings.at(0).name, "SceneBuffer");
  ASSERT_EQ(d.bindings.at(1).type, vk::DescriptorType::eUniformBuffer);
  ASSERT_EQ(d.bindings.at(1).name, "ShadowBuffer");

  ASSERT_EQ(d.bindings.at(2).type, vk::DescriptorType::eCombinedImageSampler);
  ASSERT_EQ(d.bindings.at(2).name, "samplerPointLightDepths");
  ASSERT_EQ(d.bindings.at(2).dim, 1);

  ASSERT_EQ(d.bindings.at(3).type, vk::DescriptorType::eCombinedImageSampler);
  ASSERT_EQ(d.bindings.at(3).name, "samplerDirectionalLightDepths");
  ASSERT_EQ(d.bindings.at(3).dim, 1);

  ASSERT_EQ(d.bindings.at(4).type, vk::DescriptorType::eCombinedImageSampler);
  ASSERT_EQ(d.bindings.at(4).name, "samplerTexturedLightDepths");
  ASSERT_EQ(d.bindings.at(4).dim, 1);

  ASSERT_EQ(d.bindings.at(5).type, vk::DescriptorType::eCombinedImageSampler);
  ASSERT_EQ(d.bindings.at(5).name, "samplerSpotLightDepths");
  ASSERT_EQ(d.bindings.at(5).dim, 1);

  // camera
  d = desc[1];
  ASSERT_EQ(d.type, UniformBindingType::eCamera);
  ASSERT_EQ(d.bindings.at(0).type, vk::DescriptorType::eUniformBuffer);
  ASSERT_EQ(d.bindings.at(0).name, "CameraBuffer");

  auto buffer = d.buffers[d.bindings.at(0).arrayIndex];
  ASSERT_TRUE(buffer->elements.contains("viewMatrix"));
  ASSERT_TRUE(buffer->elements.contains("viewMatrixInverse"));
  ASSERT_TRUE(buffer->elements.contains("projectionMatrix"));
  ASSERT_TRUE(buffer->elements.contains("projectionMatrixInverse"));
  ASSERT_TRUE(buffer->elements.contains("prevViewMatrix"));
  ASSERT_TRUE(buffer->elements.contains("prevViewMatrixInverse"));
  ASSERT_TRUE(buffer->elements.contains("width"));
  ASSERT_TRUE(buffer->elements.contains("height"));
}
