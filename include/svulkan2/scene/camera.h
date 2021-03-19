#pragma once
#include "node.h"
#include "svulkan2/common/config.h"
#include <vector>

namespace svulkan2 {
namespace scene {

class Camera : public Node {
public:
  enum Type {
    eOrthographic,
    ePerspective,
    eFullPerspective,
    eMatrix,
    eUndefined
  };

private:
  glm::mat4 mProjectionMatrix{1};

  // shared by all cameras
  float mNear{0.05};
  float mFar{10};
  float mAspect{1};

  // used by perspective
  float mFovy{glm::radians(45.f)};

  // used by orthographic
  float mScaling{1};

  // used by full
  float mFx{0};
  float mFy{0};
  float mCx{0};
  float mCy{0};
  float mSkew{0};

  Camera::Type mType{eUndefined};

public:
  Camera(std::string const &name = "");
  // TODO: test all these functions
  void setFullPerspectiveParameters(float near, float far, float fx, float fy,
                                    float cx, float cy, float width,
                                    float height, float skew);

  void setIntrinsicMatrix(glm::mat3 const &intrinsic, float near, float far,
                          float width, float height);

  void setPerspectiveParameters(float near, float far, float fovy,
                                float aspect);

  void setOrthographicParameters(float near, float far, float aspect,
                                 float scaling);

  float getNear() const;
  float getFar() const;
  float getAspect() const;
  float getFovy() const;
  float getOrthographicScaling() const;

  float getFx() const;
  float getFy() const;
  float getCx() const;
  float getCy() const;
  float getSkew() const;

  Camera::Type getCameraType() const { return mType; };

  void uploadToDevice(core::Buffer &cameraBuffer, uint32_t width,
                      uint32_t height, StructDataLayout const &cameraLayout);

  inline glm::mat4 getProjectionMatrix() const { return mProjectionMatrix; }
};

} // namespace scene
} // namespace svulkan2
