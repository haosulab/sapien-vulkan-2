#pragma once
#include "node.h"
#include "svulkan2/common/config.h"
#include <vector>

namespace svulkan2 {
namespace scene {

class PointLight : public Node {
  glm::vec4 mColor{0, 0, 0, 1};
  bool mCastShadow{};

public:
  PointLight(std::string const &name = "");
  inline void setColor(glm::vec4 const &color) { mColor = color; }
  inline glm::vec4 getColor() const { return mColor; }
  inline void enableShadow(bool enable) { mCastShadow = enable; }
  inline bool isShadowEnabled() const { return mCastShadow; }
};

class DirectionalLight : public Node {
  glm::vec4 mColor{0, 0, 0, 1};
  bool mCastShadow{};

public:
  DirectionalLight(std::string const &name = "");
  inline void setColor(glm::vec4 const &color) { mColor = color; }
  inline glm::vec4 getColor() const { return mColor; }
  inline void enableShadow(bool enable) { mCastShadow = enable; }
  inline bool isShadowEnabled() const { return mCastShadow; }

  void setDirection(glm::vec3 const &dir);
};

} // namespace scene
} // namespace svulkan2
