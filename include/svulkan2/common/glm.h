#pragma once
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_RADIANS

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

namespace svulkan2 {
namespace math {

inline glm::mat4 ortho(float left, float right, float top, float bottom, float near,
                float far) {
  float tx = -(right + left) / (right - left);
  float ty = -(bottom + top) / (bottom - top);
  float tz = -near / (far - near);

  return glm::mat4(2 / (right - left), 0, 0, 0, 0, 2 / (bottom - top), 0, 0, 0,
                   0, -1 / (far - near), 0, tx, ty, tz, 1);
}

inline glm::mat4 perspective(float fovy, float aspect, float near, float far) {
  float f = 1 / glm::tan(fovy / 2);
  return glm::mat4(f / aspect, 0, 0, 0, 0, -f, 0, 0, 0, 0, -far / (far - near), -1, 0, 0,
                   -far * near / (far - near), 0);
}

} // namespace math
} // namespace svulkan2
