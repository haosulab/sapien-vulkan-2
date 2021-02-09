#pragma once
#include "svulkan2/common/config.h"
#include "svulkan2/core/allocator.h"
#include "svulkan2/resource/model.h"
#include <vector>

namespace svulkan2 {
namespace scene {
class Node;
}

namespace resource {

class SVCamera {
  // std::shared_ptr<StructDataLayout> mBufferLayout;
  // std::vector<char> mBuffer;
  scene::Node *mParentNode;

  // std::unique_ptr<core::Buffer> mDeviceBuffer;

  glm::mat4 mPrevModelMatrix{1};
  glm::mat4 mModelMatrix{1};
  glm::mat4 mProjectionMatrix{1};

  float mNear{0.05};
  float mFar{10};
  float mFovy{glm::radians(45.f)};
  float mAspect{1};
  float mScaling{1};
  bool mIsOrtho = false;

public:
  SVCamera();
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

  inline void setParentNode(scene::Node *node) { mParentNode = node; };

  // void createDeviceResources(core::Context &context);
  void uploadToDevice(core::Buffer &cameraBuffer,
                      StructDataLayout const &cameraLayout);

  void setPrevModelMatrix(glm::mat4 const &matrix);
  void setModelMatrix(glm::mat4 const &matrix);

  inline glm::mat4 getPrevModelMatrix() const { return mPrevModelMatrix; };
  inline glm::mat4 getModelMatrix() const { return mModelMatrix; };
  inline glm::mat4 getPrevViewMatrix() const {
    return glm::affineInverse(mPrevModelMatrix);
  }
  inline glm::mat4 getViewMatrix() const {
    return glm::affineInverse(mModelMatrix);
  }
  inline glm::mat4 getProjectionMatrix() const { return mProjectionMatrix; }
};

} // namespace resource
} // namespace svulkan2
