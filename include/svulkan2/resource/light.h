#pragma once
#include "svulkan2/common/glm.h"

namespace svulkan2 {

namespace resource {

struct PointLight {
  glm::vec4 position;
  glm::vec4 color;
};

struct DirectionalLight {
  glm::vec4 direction;
  glm::vec4 color;
};

} // namespace resource
} // namespace svulkan2
