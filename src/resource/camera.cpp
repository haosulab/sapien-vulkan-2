#include "svulkan2/resource/camera.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace resource {

SVCamera::SVCamera(std::shared_ptr<StructDataLayout> bufferLayout)
    : mBufferLayout(bufferLayout), mBuffer(bufferLayout->size) {
  mViewMatrixOffset = mBufferLayout->elements.at("viewMatrix").offset;
  mViewMatrixInverseOffset =
      mBufferLayout->elements.at("viewMatrixInverse").offset;
  mProjectionMatrixOffset =
      mBufferLayout->elements.at("projectionMatrix").offset;
  mProjectionMatrixInverseOffset =
      mBufferLayout->elements.at("projectionMatrixInverse").offset;

  auto it = mBufferLayout->elements.find("prevViewMatrix");
  if (it != mBufferLayout->elements.end()) {
    mPrevViewMatrixOffset = it->second.offset;
    mPrevViewMatrixInverseOffset =
        mBufferLayout->elements.at("prevViewMatrixInverse").offset;
  }
}

void SVCamera::setPerspectiveParameters(float near, float far, float fovy,
                                      float aspect) {
  mDirty = true;
  mNear = near;
  mFar = far;
  mFovy = fovy;
  mAspect = aspect;
  mProjectionMatrix = glm::perspective(fovy, aspect, near, far);
  mProjectionMatrix[1][1] *= -1;

  auto projInverse = glm::inverse(mProjectionMatrix);
  std::memcpy(mBuffer.data() + mProjectionMatrixOffset,
              &mProjectionMatrix[0][0], 64);
  std::memcpy(mBuffer.data() + mProjectionMatrixInverseOffset,
              &projInverse[0][0], 64);
}

void SVCamera::setOrthographicParameters(float near, float far, float aspect,
                                       float scaling) {
  mDirty = true;
  mNear = near;
  mFar = far;
  mAspect = aspect;
  mScaling = scaling;
  mProjectionMatrix = glm::ortho(-scaling * aspect, scaling * aspect, -scaling,
                                 scaling, near, far);
  mProjectionMatrix[1][1] *= -1;
  auto projInverse = glm::inverse(mProjectionMatrix);
  std::memcpy(mBuffer.data() + mProjectionMatrixOffset,
              &mProjectionMatrix[0][0], 64);
  std::memcpy(mBuffer.data() + mProjectionMatrixInverseOffset,
              &projInverse[0][0], 64);
}

void SVCamera::createDeviceResources(core::Context &context) {
  mDirty = true;
  mDeviceBuffer =
      context.getAllocator().allocateUniformBuffer(mBufferLayout->size);
}

void SVCamera::uploadToDevice() {
  if (!mDeviceBuffer) {
    std::runtime_error("device resources have not been created");
  }
  if (mDirty) {
    mDeviceBuffer->upload(mBuffer);
    mDirty = false;
  }
}

void SVCamera::setPrevModelMatrix(glm::mat4 const &matrix) {
  mDirty = true;
  mPrevModelMatrix = matrix;
  if (mPrevViewMatrixOffset >= 0) {
    auto viewMatrix = glm::affineInverse(matrix);
    std::memcpy(mBuffer.data() + mPrevViewMatrixInverseOffset, &matrix[0][0],
                64);
    std::memcpy(mBuffer.data() + mPrevViewMatrixOffset, &viewMatrix[0][0], 64);
  }
}

void SVCamera::setModelMatrix(glm::mat4 const &matrix) {
  mDirty = true;
  mModelMatrix = matrix;
  auto viewMatrix = glm::affineInverse(matrix);
  std::memcpy(mBuffer.data() + mViewMatrixInverseOffset, &matrix[0][0], 64);
  std::memcpy(mBuffer.data() + mViewMatrixOffset, &viewMatrix[0][0], 64);
}

} // namespace resource
} // namespace svulkan2
