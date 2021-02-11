#include "svulkan2/scene/scene.h"
#include <algorithm>

namespace svulkan2 {
namespace scene {

Scene::Scene() {
  mNodes.push_back(std::make_unique<Node>());
  mRootNode = mNodes.back().get();
  mRootNode->setScene(this);
}

Node &Scene::addNode(Node &parent) {
  mNodes.push_back(std::make_unique<Node>());
  mNodes.back()->setScene(this);
  mNodes.back()->setParent(parent);
  parent.addChild(*mNodes.back());
  return *mNodes.back();
}

Node &Scene::addNode() { return addNode(getRootNode()); }

void Scene::removedNode(Node &node) {
  node.markRemoved();
  node.getParent().removeChild(node);
  for (Node *c : node.getChildren()) {
    node.getParent().addChild(*c);
  }
  // TODO: fix child transform here?
}

void Scene::prepareObjectCameraForRendering() {
  mRootNode->updateGlobalModelMatrixRecursive();
  mRootNode->updateObjectCameraModelMatrixRecursive();
}

void Scene::clearNodes() { mNodes.resize(1); }

void Scene::forceRemove() {
  mNodes.erase(std::remove_if(mNodes.begin(), mNodes.end(),
                              [](std::unique_ptr<Node> &node) {
                                return node->isMarkedRemoved();
                              }),
               mNodes.end());
}

std::vector<std::shared_ptr<resource::SVObject>> Scene::getObjects() const {
  return mRootNode->getObjectsRecursive();
}

} // namespace scene
} // namespace svulkan2
