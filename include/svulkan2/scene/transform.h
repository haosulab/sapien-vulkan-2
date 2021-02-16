#pragma once
#include "svulkan2/common/glm.h"

namespace svulkan2 {
namespace scene {

struct Transform {
  glm::vec3 position{0, 0, 0};
  glm::quat rotation{glm::quat(1, 0, 0, 0)};
  glm::vec3 scale{1, 1, 1};

  glm::mat4 prevWorldModelMatrix;
  glm::mat4 worldModelMatrix;
};

} // namespace scene
} // namespace svulkan2
