#include "svulkan2/scene/node.h"
#include <algorithm>

namespace svulkan2 {
namespace scene {

Node::Node(std::string const &name) : mName(name) {}

void Node::removeChild(Node &child) {
  auto it = std::find(mChildren.begin(), mChildren.end(), &child);
  if (it == mChildren.end()) {
    throw std::invalid_argument("child node does not exist");
  }
  mChildren.erase(it);
}

void Node::clearChild() { mChildren = {}; }

void Node::setObject(std::shared_ptr<resource::SVObject> object) {
  mObject = object;
  mObject->setParentNode(this);
}

std::shared_ptr<resource::SVObject> Node::removeObject() {
  mObject->setParentNode(nullptr);
  auto obj = mObject;
  mObject = nullptr;
  return obj;
}

void Node::setCamera(std::shared_ptr<resource::SVCamera> camera) {
  mCamera = camera;
  mCamera->setParentNode(this);
}

std::shared_ptr<resource::SVCamera> Node::removeCamera() {
  mCamera->setParentNode(nullptr);
  auto obj = mCamera;
  mCamera = nullptr;
  return obj;
}

void Node::setTransform(Transform const &transform) { mTransform = transform; }

void Node::updateGlobalModelMatrixRecursive() {
  glm::mat4 localMatrix = glm::translate(glm::mat4(1), mTransform.position) *
                          glm::toMat4(mTransform.rotation) *
                          glm::scale(glm::mat4(1), mTransform.scale);
  if (!mParent) {
    mTransform.worldModelMatrix = localMatrix;
  }
  mTransform.worldModelMatrix =
      mParent->mTransform.worldModelMatrix * localMatrix;
  for (auto c : mChildren) {
    c->updateGlobalModelMatrixRecursive();
  }
}

void Node::updateObjectCameraModelMatrixRecursive() {
  if (mObject) {
    mObject->setPrevModelMatrix(mObject->getModelMatrix());
    mObject->setModelMatrix(mTransform.worldModelMatrix);
  }
  if (mCamera) {
    mCamera->setPrevModelMatrix(mCamera->getModelMatrix());
    mCamera->setModelMatrix(mTransform.worldModelMatrix);
  }
  for (auto c : mChildren) {
    c->updateObjectCameraModelMatrixRecursive();
  }
}

std::vector<std::shared_ptr<resource::SVObject>>
Node::getObjectsRecursive() const {
  std::vector<std::shared_ptr<resource::SVObject>> result;
  if (mObject) {
    result.push_back(mObject);
  }
  for (auto c : mChildren) {
    auto childObjects = c->getObjectsRecursive();
    result.insert(result.end(), childObjects.begin(), childObjects.end());
  }
  return result;
}

} // namespace scene
} // namespace svulkan2
