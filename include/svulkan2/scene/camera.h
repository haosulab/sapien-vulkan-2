#pragma once
#include "node.h"
#include "svulkan2/common/config.h"
#include <vector>

namespace svulkan2 {
namespace scene {

class Camera : public Node {
  glm::mat4 mProjectionMatrix{1};

  float mNear{0.05};
  float mFar{10};
  float mFovy{glm::radians(45.f)};
  float mAspect{1};
  float mScaling{1};
  bool mIsOrtho = false;

public:
  Camera(std::string const &name = "");
  void setCameraParameters(float near, float far, float fx, float fy, float cx,
                           float cy, float width, float height);
  void setPerspectiveParameters(float near, float far, float fovy,
                                float aspect);
  void setOrthographicParameters(float near, float far, float aspect,
                                 float scaling);

  inline float getNear() { return mNear; }
  inline float getFar() { return mFar; }
  inline float getAspect() { return mAspect; }
  inline float getFovy() { return mFovy; }
  inline float getOrthographicScaling() { return mScaling; }
  inline float isOrthographic() { return mIsOrtho; }

  void uploadToDevice(core::Buffer &cameraBuffer, uint32_t width,
                      uint32_t height, StructDataLayout const &cameraLayout);

  inline glm::mat4 getProjectionMatrix() const { return mProjectionMatrix; }
};

} // namespace scene
} // namespace svulkan2
