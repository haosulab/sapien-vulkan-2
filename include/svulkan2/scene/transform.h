#pragma once
#include "svulkan2/common/glm.h"

namespace svulkan2 {
namespace scene {

struct Transform {
  glm::vec3 position{0, 0, 0};
  glm::quat rotation{glm::quat(1, 0, 0, 0)};
  glm::vec3 scale{1, 1, 1};

  static Transform FromMatrix(glm::mat4 const &mat) {
    Transform T;
    glm::vec3 skew;
    glm::vec4 pers;
    glm::decompose(mat, T.scale, T.rotation, T.position, skew, pers);
    return T;
  }

  glm::mat4 matrix() const {
    return glm::translate(glm::mat4(1), position) * glm::toMat4(rotation) *
           glm::scale(glm::mat4(1), scale);
  }
};

struct TransformWithCache : public Transform {
  glm::mat4 prevWorldModelMatrix{};
  glm::mat4 worldModelMatrix{};

  TransformWithCache() {}

  TransformWithCache(Transform const &other) {
    position = other.position;
    rotation = other.rotation;
    scale = other.scale;
  }

  TransformWithCache &operator=(Transform const &other) {
    position = other.position;
    rotation = other.rotation;
    scale = other.scale;
    return *this;
  }
};

} // namespace scene
} // namespace svulkan2
