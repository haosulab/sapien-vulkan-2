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

} // namespace scene
} // namespace svulkan2
