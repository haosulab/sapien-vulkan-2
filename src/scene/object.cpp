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
  for (auto &[name, value] : mCustomData) {
    if (objectLayout.elements.contains(name)) {
      auto &elem = objectLayout.elements.at(name);
      if (elem.dtype != value.dtype) {
        throw std::runtime_error("Upload object failed: object attribute \"" +
                                 name + "\" does not match declared type.");
        std::memcpy(buffer.data() + elem.offset, &value.floatValue, elem.size);
      }
    }
  }
  if (objectLayout.elements.contains("transparency")) {
    auto &elem = objectLayout.elements.at("transparency");
    if (elem.dtype != DataType::eFLOAT) {
      throw std::runtime_error("Upload object failed: object attribute "
                               "\"transparency\" must be a float");
    }
    std::memcpy(buffer.data() + elem.offset, &mTransparency, 4);
  }
  objectBuffer.upload(buffer);
}

void Object::setCustomDataFloat(std::string const &name, float x) {
  mCustomData[name] = {.dtype = DataType::eFLOAT, .floatValue = x};
}
void Object::setCustomDataFloat2(std::string const &name, glm::vec2 x) {
  mCustomData[name] = {.dtype = DataType::eFLOAT2, .float2Value = x};
}
void Object::setCustomDataFloat3(std::string const &name, glm::vec3 x) {
  mCustomData[name] = {.dtype = DataType::eFLOAT3, .float3Value = x};
}
void Object::setCustomDataFloat4(std::string const &name, glm::vec4 x) {
  mCustomData[name] = {.dtype = DataType::eFLOAT4, .float4Value = x};
}
void Object::setCustomDataFloat44(std::string const &name, glm::mat4 x) {
  mCustomData[name] = {.dtype = DataType::eFLOAT44, .float44Value = x};
}
void Object::setCustomDataInt(std::string const &name, int x) {
  mCustomData[name] = {.dtype = DataType::eINT, .intValue = x};
}
void Object::setCustomDataInt2(std::string const &name, glm::ivec2 x) {
  mCustomData[name] = {.dtype = DataType::eINT2, .int2Value = x};
}
void Object::setCustomDataInt3(std::string const &name, glm::ivec3 x) {
  mCustomData[name] = {.dtype = DataType::eINT3, .int3Value = x};
}
void Object::setCustomDataInt4(std::string const &name, glm::ivec4 x) {
  mCustomData[name] = {.dtype = DataType::eINT4, .int4Value = x};
}

} // namespace scene
} // namespace svulkan2