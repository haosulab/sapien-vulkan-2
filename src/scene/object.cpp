#include "svulkan2/scene/object.h"

namespace svulkan2 {
namespace scene {

Object::Object(std::shared_ptr<resource::SVModel> model,
               std::string const &name)
    : Node(name), mModel(model) {}

void Object::uploadToDevice(core::Buffer &objectBuffer,
                            StructDataLayout const &objectLayout) {
  std::vector<char> buffer(objectLayout.size);
  std::memcpy(buffer.data() + objectLayout.elements.at("modelMatrix").offset,
              &mTransform.worldModelMatrix[0][0], 64);
  std::memcpy(buffer.data() + objectLayout.elements.at("segmentation").offset,
              &mSegmentation[0], 16);
  auto it = objectLayout.elements.find("prevModelMatrix");
  if (it != objectLayout.elements.end()) {
    std::memcpy(buffer.data() +
                    objectLayout.elements.at("prevModelMatrix").offset,
                &mTransform.prevWorldModelMatrix[0][0], 64);
  }
  objectBuffer.upload(buffer);
}

} // namespace scene
} // namespace svulkan2
