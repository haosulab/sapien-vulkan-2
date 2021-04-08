#pragma once
#include "node.h"
#include "svulkan2/common/config.h"
#include <vector>

namespace svulkan2 {
namespace scene {

class DirectionalLight : public Node {
  glm::vec4 mColor{0, 0, 0, 1};
  bool mCastShadow{};
  float mShadowNear{0};
  float mShadowFar{10};
  float mShadowScaling{10};

public:
  DirectionalLight(std::string const &name = "");
  inline void setColor(glm::vec4 const &color) { mColor = color; }
  inline glm::vec4 getColor() const { return mColor; }
  void enableShadow(bool enable);
  inline bool isShadowEnabled() const { return mCastShadow; }
  void setShadowParameters(float near, float far, float scaling);
  glm::vec3 getDirection() const;
  void setDirection(glm::vec3 const &dir);

  glm::mat4 getShadowProjectionMatrix() const;
};

class PointLight : public Node {
  glm::vec4 mColor{0, 0, 0, 1};
  bool mCastShadow{};
  float mShadowNear{0};
  float mShadowFar{10};

public:
  static std::array<glm::mat4, 6> getModelMatrices(glm::vec3 const &center);

  PointLight(std::string const &name = "");
  inline void setColor(glm::vec4 const &color) { mColor = color; }
  inline glm::vec4 getColor() const { return mColor; }
  void enableShadow(bool enable);
  inline bool isShadowEnabled() const { return mCastShadow; }
  void setShadowParameters(float near, float far);
  glm::mat4 getShadowProjectionMatrix() const;
};

class CustomLight : public Node {
  glm::mat4 mProjectionMatrix{1};

public:
  CustomLight(std::string const &name = "");
  inline void setShadowProjectionMatrix(glm::mat4 const &mat) {
    mProjectionMatrix = mat;
  }
  inline glm::mat4 getShadowProjectionMatrix() const {
    return mProjectionMatrix;
  };
};

class SpotLight : public Node {
  glm::vec4 mColor{0, 0, 0, 1};
  bool mCastShadow{};
  float mShadowNear{0};
  float mShadowFar{10};
  float mFov = 1.5708;

public:
  SpotLight(std::string const &name = "");
  inline void setColor(glm::vec4 const &color) { mColor = color; }
  inline glm::vec4 getColor() const { return mColor; }
  void enableShadow(bool enable);
  inline bool isShadowEnabled() const { return mCastShadow; }
  void setShadowParameters(float near, float far);
  void setDirection(glm::vec3 const &dir);
  glm::vec3 getDirection() const;
  void setFov(float fov);
  float getFov() const;

  glm::mat4 getShadowProjectionMatrix() const;
};

} // namespace scene
} // namespace svulkan2
