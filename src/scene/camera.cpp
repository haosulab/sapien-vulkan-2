#include "svulkan2/scene/camera.h"
#include <vector>

namespace svulkan2 {
namespace scene {

Camera::Camera(std::string const &name) : Node(name) {}

void Camera::setCameraParameters(float near, float far, float fx, float fy,
                                 float cx, float cy, float width,
                                 float height) {
  mNear = near;
  mFar = far;
  glm::mat4 mat(1);
  mat[0][0] = (2.f * fx) / width;
  mat[1][1] = (2.f * fy) / height;
  // depth [0,1], for depth [-1,1], it is -(far+near)/(far-near)
  mat[2][2] = -far / (far - near);
  mat[3][2] = -2.f * far * near / (far - near);
  mat[2][3] = -1.f;
  mat[2][0] = (2.f * cx) / width - 1;
  mat[2][1] = (2.f * cy) / height - 1;

  // flip y
  mat[2][1] *= -1;
  mat[1][1] *= -1;

  mProjectionMatrix = mat;
}

void Camera::setPerspectiveParameters(float near, float far, float fovy,
                                      float aspect) {
  mNear = near;
  mFar = far;
  mFovy = fovy;
  mAspect = aspect;
  mProjectionMatrix = glm::perspective(fovy, aspect, near, far);
  mProjectionMatrix[1][1] *= -1;
}

void Camera::setOrthographicParameters(float near, float far, float aspect,
                                       float scaling) {
  mNear = near;
  mFar = far;
  mAspect = aspect;
  mScaling = scaling;
  mProjectionMatrix = glm::ortho(-scaling * aspect, scaling * aspect, -scaling,
                                 scaling, near, far);
  mProjectionMatrix[1][1] *= -1;
}

void Camera::uploadToDevice(core::Buffer &cameraBuffer, uint32_t width,
                            uint32_t height,
                            StructDataLayout const &cameraLayout) {
  std::vector<char> mBuffer(cameraLayout.size);

  auto viewMatrix = glm::affineInverse(mTransform.worldModelMatrix);
  auto projInv = glm::inverse(mProjectionMatrix);
  std::memcpy(mBuffer.data() + cameraLayout.elements.at("viewMatrix").offset,
              &viewMatrix[0][0], 64);
  std::memcpy(mBuffer.data() +
                  cameraLayout.elements.at("viewMatrixInverse").offset,
              &mTransform.worldModelMatrix[0][0], 64);
  std::memcpy(mBuffer.data() +
                  cameraLayout.elements.at("projectionMatrix").offset,
              &mProjectionMatrix[0][0], 64);
  std::memcpy(mBuffer.data() +
                  cameraLayout.elements.at("projectionMatrixInverse").offset,
              &projInv[0][0], 64);

  auto it = cameraLayout.elements.find("prevViewMatrix");
  if (it != cameraLayout.elements.end()) {
    auto prevViewMatrix = glm::affineInverse(mTransform.prevWorldModelMatrix);
    std::memcpy(mBuffer.data() + it->second.offset, &prevViewMatrix[0][0], 64);
  }

  it = cameraLayout.elements.find("prevViewMatrixInverse");
  if (it != cameraLayout.elements.end()) {
    std::memcpy(mBuffer.data() +
                    cameraLayout.elements.at("prevViewMatrixInverse").offset,
                &mTransform.prevWorldModelMatrix[0][0], 64);
  }

  it = cameraLayout.elements.find("width");
  if (it != cameraLayout.elements.end()) {
    float fwidth = static_cast<float>(width);
    std::memcpy(mBuffer.data() + cameraLayout.elements.at("width").offset,
                &fwidth, sizeof(float));
  }

  it = cameraLayout.elements.find("height");
  if (it != cameraLayout.elements.end()) {
    float fheight = static_cast<float>(height);
    std::memcpy(mBuffer.data() + cameraLayout.elements.at("height").offset,
                &fheight, sizeof(float));
  }

  cameraBuffer.upload<char>(mBuffer);
}

} // namespace scene
} // namespace svulkan2
