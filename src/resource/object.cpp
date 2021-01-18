#include "svulkan2/resource/object.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace resource {

SVObject::SVObject(std::shared_ptr<StructDataLayout> bufferLayout,
                   std::shared_ptr<SVModel> model)
    : mBufferLayout(bufferLayout), mModel(model), mBuffer(bufferLayout->size) {
  auto it = mBufferLayout->elements.find("prevModelMatrix");
  if (it != mBufferLayout->elements.end()) {
    mPrevModelMatrixOffset = it->second.offset;
  }
  mModelMatrixOffset = mBufferLayout->elements.at("modelMatrix").offset;
  mSegmentationOffset = mBufferLayout->elements.at("segmentation").offset;
}

void SVObject::createDeviceResources(core::Context &context) {
  mDirty = true;
  mDeviceBuffer =
      context.getAllocator().allocateUniformBuffer(mBufferLayout->size);
}

void SVObject::uploadToDevice() {
  if (!mDeviceBuffer) {
    std::runtime_error("device resources have not been created");
  }
  if (mDirty) {
    mDeviceBuffer->upload(mBuffer);
    mDirty = false;
  }
}

void SVObject::setPrevModelMatrix(glm::mat4 const &matrix) {
  mDirty = true;
  mPrevModelMatrix = matrix;
  if (mPrevModelMatrixOffset >= 0) {
    std::memcpy(mBuffer.data() + mPrevModelMatrixOffset, &matrix[0][0], 64);
  }
}

void SVObject::setModelMatrix(glm::mat4 const &matrix) {
  mDirty = true;
  mModelMatrix = matrix;
  std::memcpy(mBuffer.data() + mModelMatrixOffset, &matrix[0][0], 64);
}

void SVObject::setSegmentation(glm::uvec4 const &segmentation) {
  mDirty = true;
  mSegmentation = segmentation;
  std::memcpy(mBuffer.data() + mSegmentationOffset, &segmentation[0], 16);
}

} // namespace resource
} // namespace svulkan2
