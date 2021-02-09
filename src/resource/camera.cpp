#include "svulkan2/resource/camera.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace resource {

SVCamera::SVCamera() {}

void SVCamera::setPerspectiveParameters(float near, float far, float fovy,
                                        float aspect) {
  mNear = near;
  mFar = far;
  mFovy = fovy;
  mAspect = aspect;
  mProjectionMatrix = glm::perspective(fovy, aspect, near, far);
  mProjectionMatrix[1][1] *= -1;
}

void SVCamera::setOrthographicParameters(float near, float far, float aspect,
                                         float scaling) {
  mNear = near;
  mFar = far;
  mAspect = aspect;
  mScaling = scaling;
  mProjectionMatrix = glm::ortho(-scaling * aspect, scaling * aspect, -scaling,
                                 scaling, near, far);
  mProjectionMatrix[1][1] *= -1;
}

void SVCamera::uploadToDevice(core::Buffer &cameraBuffer,
                              StructDataLayout const &cameraLayout) {
  std::vector<char> mBuffer(cameraLayout.size);

  auto viewMatrix = glm::affineInverse(mModelMatrix);
  auto projInv = glm::affineInverse(mProjectionMatrix);
  std::memcpy(mBuffer.data() + cameraLayout.elements.at("viewMatrix").offset,
              &viewMatrix[0][0], 64);
  std::memcpy(mBuffer.data() +
                  cameraLayout.elements.at("viewMatrixInverse").offset,
              &mModelMatrix[0][0], 64);
  std::memcpy(mBuffer.data() +
                  cameraLayout.elements.at("projectionMatrix").offset,
              &mProjectionMatrix[0][0], 64);
  std::memcpy(mBuffer.data() +
                  cameraLayout.elements.at("projectionMatrixInverse").offset,
              &projInv[0][0], 64);

  auto it = cameraLayout.elements.find("prevViewMatrix");
  if (it != cameraLayout.elements.end()) {
    auto prevViewMatrix = glm::affineInverse(mPrevModelMatrix);
    std::memcpy(mBuffer.data() + it->second.offset, &prevViewMatrix[0][0], 64);
    std::memcpy(mBuffer.data() +
                    cameraLayout.elements.at("prevViewMatrixInverse").offset,
                &mPrevModelMatrix[0][0], 64);
  }
}

void SVCamera::setPrevModelMatrix(glm::mat4 const &matrix) {
  mPrevModelMatrix = matrix;
}

void SVCamera::setModelMatrix(glm::mat4 const &matrix) {
  mModelMatrix = matrix;
}

} // namespace resource
} // namespace svulkan2
