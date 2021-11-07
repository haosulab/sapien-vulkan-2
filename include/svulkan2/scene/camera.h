#pragma once
#include "node.h"
#include "svulkan2/common/config.h"
#include <vector>

namespace svulkan2 {
namespace scene {

class Camera : public Node {
public:
  enum Type { eOrthographic, ePerspective, eMatrix, eUndefined };

private:
  glm::mat4 mProjectionMatrix{1};

  float mWidth{1};
  float mHeight{1};
  float mNear{0.01};
  float mFar{10};

  float mFx{0};
  float mFy{0};
  float mCx{0};
  float mCy{0};
  float mSkew{0};

  // used by orthographic
  float mScaling{1};

  Camera::Type mType{eUndefined};

public:
  Camera(std::string const &name = "");

  void setPerspectiveParameters(float near, float far, float fovy, float width,
                                float height);

  void setPerspectiveParameters(float near, float far, float fx, float fy,
                                float cx, float cy, float width, float height,
                                float skew);

  void setIntrinsicMatrix(glm::mat3 const &intrinsic, float near, float far,
                          float width, float height);

  void setOrthographicParameters(float near, float far, float scaling,
                                 float width, float height);

  void setWidth(float width);
  void setHeight(float height);

  float getWidth() const;
  float getHeight() const;
  float getNear() const;
  float getFar() const;
  float getFovx() const;
  float getFovy() const;
  float getOrthographicScaling() const;

  float getFx() const;
  float getFy() const;
  float getCx() const;
  float getCy() const;
  float getSkew() const;

  Camera::Type getCameraType() const { return mType; };

  void uploadToDevice(core::Buffer &cameraBuffer,
                      StructDataLayout const &cameraLayout);

  inline glm::mat4 getProjectionMatrix() const { return mProjectionMatrix; }
};

} // namespace scene
} // namespace svulkan2
