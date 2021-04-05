#include "svulkan2/scene/light.h"
#include "svulkan2/scene/scene.h"

namespace svulkan2 {
namespace scene {

std::array<glm::mat4, 6> PointLight::getModelMatrices(glm::vec3 const &center) {
  const glm::vec3 x(1, 0, 0);
  const glm::vec3 y(0, 1, 0);
  const glm::vec3 z(0, 0, 1);

  glm::mat4 m1 = glm::mat3(glm::mat3(-z, y, -x));
  m1[3] = glm::vec4(center, 1.f);

  glm::mat4 m2 = glm::mat3(glm::mat3(z, y, x));
  m2[3] = glm::vec4(center, 1.f);

  glm::mat4 m3 = glm::mat3(glm::mat3(x, -z, -y));
  m3[3] = glm::vec4(center, 1.f);

  glm::mat4 m4 = glm::mat3(glm::mat3(x, z, y));
  m4[3] = glm::vec4(center, 1.f);

  glm::mat4 m5 = glm::mat3(glm::mat3(x, y, -z));
  m5[3] = glm::vec4(center, 1.f);

  glm::mat4 m6 = glm::mat3(glm::mat3(-x, y, z));
  m6[3] = glm::vec4(center, 1.f);

  return {m1, m2, m3, m4, m5, m6};
}

PointLight::PointLight(std::string const &name) : Node(name) {}
void PointLight::enableShadow(bool enable) {
  mCastShadow = enable;
  mScene->reorderLights();
}

void PointLight::setShadowParameters(float near, float far) {
  mShadowNear = near;
  mShadowFar = far;
}

glm::mat4 PointLight::getShadowProjectionMatrix() const {
  auto mat =
      math::perspective(glm::pi<float>() / 2.f, 1.f, mShadowNear, mShadowFar);
  return mat;
}

DirectionalLight::DirectionalLight(std::string const &name) : Node(name){};

void DirectionalLight::enableShadow(bool enable) {
  mCastShadow = enable;
  mScene->reorderLights();
}

void DirectionalLight::setDirection(glm::vec3 const &dir) {
  auto z = -glm::normalize(dir);
  glm::vec3 x(1, 0, 0);
  glm::vec3 y;
  if (std::abs(glm::dot(z, x)) > 0.05) {
    y = glm::normalize(glm::cross(z, x));
  } else {
    y = glm::normalize(glm::cross(z, glm::vec3(0, 1, 0)));
  }
  x = glm::cross(y, z);
  mTransform.rotation = glm::quat(glm::mat3(x, y, z));
}

void DirectionalLight::setShadowParameters(float near, float far,
                                           float scaling) {
  mShadowNear = near;
  mShadowFar = far;
  mShadowScaling = scaling;
}

glm::mat4 DirectionalLight::getShadowProjectionMatrix() const {
  return math::ortho(-mShadowScaling, mShadowScaling, -mShadowScaling,
                     mShadowScaling, mShadowNear, mShadowFar);
}

CustomLight::CustomLight(std::string const &name) : Node(name) {}

} // namespace scene
} // namespace svulkan2
