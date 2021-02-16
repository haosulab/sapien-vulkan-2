#include "svulkan2/scene/node.h"
#include "svulkan2/scene/object.h"
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

void Node::setTransform(Transform const &transform) { mTransform = transform; }

void Node::updateGlobalModelMatrixRecursive() {
  glm::mat4 localMatrix = glm::translate(glm::mat4(1), mTransform.position) *
                          glm::toMat4(mTransform.rotation) *
                          glm::scale(glm::mat4(1), mTransform.scale);
  mTransform.prevWorldModelMatrix = mTransform.worldModelMatrix;
  if (!mParent) {
    mTransform.worldModelMatrix = localMatrix;
  } else {
    mTransform.worldModelMatrix =
        mParent->mTransform.worldModelMatrix * localMatrix;
  }
  for (auto c : mChildren) {
    c->updateGlobalModelMatrixRecursive();
  }
}

std::vector<Object *> Node::getObjectsRecursive() const {
  std::vector<Object *> result;
  for (auto c : mChildren) {
    if (auto object = dynamic_cast<Object *>(c))
      if (object) {
        result.push_back(object);
      }
    auto childObjects = c->getObjectsRecursive();
    result.insert(result.end(), childObjects.begin(), childObjects.end());
  }
  return result;
}

} // namespace scene
} // namespace svulkan2
