#include "svulkan2/scene/light.h"

namespace svulkan2 {
namespace scene {

PointLight::PointLight(std::string const &name) : Node(name) {}
DirectionalLight::DirectionalLight(std::string const &name) : Node(name){};

void DirectionalLight::setDirection(glm::vec3 const &dir) {
  auto x = glm::normalize(dir);
  glm::vec3 y(1, 0, 0);
  glm::vec3 z;
  if (glm::dot(x, y) > 0.05) {
    z = glm::normalize(glm::cross(x, y));
  } else {
    z = glm::normalize(glm::cross(x, glm::vec3(0, 1, 0)));
  }
  y = glm::cross(z, x);
  mTransform.rotation = glm::quat(glm::mat3(x, y, z));
}

} // namespace scene
} // namespace svulkan2
