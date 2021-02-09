#include "svulkan2/resource/object.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace resource {

SVObject::SVObject(std::shared_ptr<SVModel> model) : mModel(model) {}

void SVObject::uploadToDevice(core::Buffer &objectBuffer,
                              StructDataLayout const &objectLayout) {
  std::vector<char> buffer(objectLayout.size);
  std::memcpy(buffer.data() + objectLayout.elements.at("modelMatrix").offset,
              &mModelMatrix[0][0], 64);
  std::memcpy(buffer.data() + objectLayout.elements.at("segmentation").offset,
              &mSegmentation[0], 16);
  auto it = objectLayout.elements.find("prevModelMatrix");
  if (it != objectLayout.elements.end()) {
    std::memcpy(buffer.data() +
                    objectLayout.elements.at("prevModelMatrix").offset,
                &mPrevModelMatrix[0], 64);
  }
  objectBuffer.upload(buffer);
}

void SVObject::setPrevModelMatrix(glm::mat4 const &matrix) {
  mPrevModelMatrix = matrix;
}

void SVObject::setModelMatrix(glm::mat4 const &matrix) {
  mModelMatrix = matrix;
}

void SVObject::setSegmentation(glm::uvec4 const &segmentation) {
  mSegmentation = segmentation;
}

} // namespace resource
} // namespace svulkan2
