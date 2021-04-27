#include "svulkan2/scene/camera.h"
#include <vector>

namespace svulkan2 {
namespace scene {

Camera::Camera(std::string const &name) : Node(name) {}

void Camera::setFullPerspectiveParameters(float near, float far, float fx,
                                          float fy, float cx, float cy,
                                          float width, float height,
                                          float skew) {
  mType = Camera::Type::eFullPerspective;
  mNear = near;
  mFar = far;
  mAspect = width / height;

  mFx = fx;
  mFy = fy;
  mCx = cx;
  mCy = cy;
  mSkew = skew;

  glm::mat4 mat(1);
  mat[0][0] = (2.f * fx) / width;
  mat[1][1] = -(2.f * fy) / height;

  mat[2][2] = -far / (far - near);
  mat[3][2] = -far * near / (far - near);
  mat[2][3] = -1.f;
  mat[2][0] = -2.f * cx / width + 1;
  mat[2][1] = -2.f * cy / height + 1;
  mat[3][3] = 0.f;

  mat[1][0] = -2 * skew / width;

  mProjectionMatrix = mat;
}

void Camera::setIntrinsicMatrix(glm::mat3 const &intrinsic, float near,
                                float far, float width, float height) {
  setFullPerspectiveParameters(near, far, intrinsic[0][0], intrinsic[1][1],
                               intrinsic[2][0], intrinsic[2][1], width, height,
                               intrinsic[1][0]);
}

void Camera::setPerspectiveParameters(float near, float far, float fovy,
                                      float aspect) {
  mType = Camera::Type::ePerspective;

  mNear = near;
  mFar = far;
  mFovy = fovy;
  mAspect = aspect;
  mProjectionMatrix = math::perspective(fovy, aspect, near, far);
}

void Camera::setOrthographicParameters(float near, float far, float aspect,
                                       float scaling) {
  mType = Camera::Type::eOrthographic;

  mNear = near;
  mFar = far;
  mAspect = aspect;
  mScaling = scaling;
  mProjectionMatrix = math::ortho(-scaling * aspect, scaling * aspect, -scaling,
                                  scaling, near, far);
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

float Camera::getNear() const {
  if (mType == Camera::Type::eUndefined) {
    throw std::runtime_error("Camera is not initialized with parameters.");
  }
  if (mType == Camera::Type::eMatrix) {
    throw std::runtime_error(
        "Camera initialized by projection matrix does not have this property");
  }

  return mNear;
}

float Camera::getFar() const {
  if (mType == Camera::Type::eUndefined) {
    throw std::runtime_error("Camera is not initialized with parameters.");
  }
  if (mType == Camera::Type::eMatrix) {
    throw std::runtime_error(
        "Camera initialized by projection matrix does not have this property");
  }
  return mFar;
}

float Camera::getAspect() const {
  if (mType == Camera::Type::eUndefined) {
    throw std::runtime_error("Camera is not initialized with parameters.");
  }
  if (mType == Camera::Type::eMatrix) {
    throw std::runtime_error(
        "Camera initialized by projection matrix does not have this property");
  }
  return mAspect;
}

float Camera::getFovy() const {
  if (mType == Camera::Type::eUndefined) {
    throw std::runtime_error("Camera is not initialized with parameters.");
  }
  if (mType == Camera::Type::eMatrix) {
    throw std::runtime_error(
        "Camera initialized by projection matrix does not have this property");
  }
  return mFovy;
}

float Camera::getOrthographicScaling() const {
  if (mType == Camera::Type::eUndefined) {
    throw std::runtime_error("Camera is not initialized with parameters.");
  }
  if (mType != eOrthographic) {
    throw std::runtime_error("Only orthographic camera has this property.");
  }
  return mScaling;
}

float Camera::getFx() const {
  if (mType == Camera::Type::eUndefined) {
    throw std::runtime_error("Camera is not initialized with parameters.");
  }
  if (mType != Camera::Type::eFullPerspective) {
    throw std::runtime_error("Only camera created by full intrinsic matrix "
                             "properties has this property.");
  }

  return mFx;
}

float Camera::getFy() const {
  if (mType == Camera::Type::eUndefined) {
    throw std::runtime_error("Camera is not initialized with parameters.");
  }
  if (mType != Camera::Type::eFullPerspective) {
    throw std::runtime_error("Only camera created by full intrinsic matrix "
                             "properties has this property.");
  }
  return mFy;
}
float Camera::getCx() const {
  if (mType == Camera::Type::eUndefined) {
    throw std::runtime_error("Camera is not initialized with parameters.");
  }
  if (mType != Camera::Type::eFullPerspective) {
    throw std::runtime_error("Only camera created by full intrinsic matrix "
                             "properties has this property.");
  }
  return mCx;
}
float Camera::getCy() const {
  if (mType == Camera::Type::eUndefined) {
    throw std::runtime_error("Camera is not initialized with parameters.");
  }
  if (mType != Camera::Type::eFullPerspective) {
    throw std::runtime_error("Only camera created by full intrinsic matrix "
                             "properties has this property.");
  }
  return mCy;
}

float Camera::getSkew() const {
  if (mType == Camera::Type::eUndefined) {
    throw std::runtime_error("Camera is not initialized with parameters.");
  }
  if (mType != Camera::Type::eFullPerspective) {
    throw std::runtime_error("Only camera created by full intrinsic matrix "
                             "properties has this property.");
  }
  return mSkew;
}

} // namespace scene
} // namespace svulkan2
