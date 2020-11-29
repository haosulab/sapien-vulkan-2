#include "svulkan2/scene/scene.h"
#include <algorithm>

namespace svulkan2 {
Scene::Scene() {
  mNodes.push_back(std::make_unique<Node>());
  mRootNode = mNodes.back().get();
  mRootNode->setScene(this);
}

void Scene::addNode(std::unique_ptr<Node> node) {
  addNode(std::move(node), getRootNode());
}

void Scene::addNode(std::unique_ptr<Node> node, Node &parent) {
  if (node == nullptr) {
    throw std::invalid_argument("addNode: nullptr is not allowed");
  }
  if (parent.getScene() != this) {
    throw std::invalid_argument("addNode: parent node is not in the scene");
  }
  node->setScene(this);
  node->setParent(parent);
  parent.addChild(*node.get());
  mNodes.push_back(std::move(node));
}

void Scene::removedNode(Node &node) {
  node.markRemoved();
  node.getParent().removeChild(node);
  for (Node *c : node.getChildren()) {
    node.getParent().addChild(*c);
  }
  // TODO: fix child transform here?
}

void Scene::clearNodes() { mNodes.resize(1); }

void Scene::forceRemove() {
  mNodes.erase(std::remove_if(mNodes.begin(), mNodes.end(),
                              [](std::unique_ptr<Node> &node) {
                                return node->isMarkedRemoved();
                              }),
               mNodes.end());
}

} // namespace svulkan2
