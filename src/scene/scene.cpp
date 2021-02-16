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
  forceRemove();
  mNodes.push_back(std::make_unique<Node>());
  mNodes.back()->setScene(this);
  mNodes.back()->setParent(parent);
  parent.addChild(*mNodes.back());
  return *mNodes.back();
}

Node &Scene::addNode() { return addNode(getRootNode()); }

Object &Scene::addObject(std::shared_ptr<resource::SVModel> model) {
  return addObject(getRootNode(), model);
}

Object &Scene::addObject(Node &parent,
                         std::shared_ptr<resource::SVModel> model) {
  forceRemove();
  auto obj = std::make_unique<Object>(model);
  auto &result = *obj;
  mNodes.push_back(std::move(obj));
  mNodes.back()->setScene(this);
  mNodes.back()->setParent(parent);
  parent.addChild(*mNodes.back());
  return result;
}

Camera &Scene::addCamera() { return addCamera(getRootNode()); }
Camera &Scene::addCamera(Node &parent) {
  forceRemove();
  auto cam = std::make_unique<Camera>();
  auto &result = *cam;
  mNodes.push_back(std::move(cam));
  mNodes.back()->setScene(this);
  mNodes.back()->setParent(parent);
  parent.addChild(*mNodes.back());
  return result;
}

void Scene::removeNode(Node &node) {
  mRequireForceRemove = true;
  node.markRemoved();
  node.getParent().removeChild(node);
  for (Node *c : node.getChildren()) {
    node.getParent().addChild(*c);
  }
}

void Scene::clearNodes() { mNodes.resize(1); }

void Scene::forceRemove() {
  if (!mRequireForceRemove) {
    return;
  }
  mNodes.erase(std::remove_if(mNodes.begin(), mNodes.end(),
                              [](std::unique_ptr<Node> &node) {
                                return node->isMarkedRemoved();
                              }),
               mNodes.end());
  mRequireForceRemove = false;
}

std::vector<Object *> Scene::getObjects() {
  forceRemove();
  return mRootNode->getObjectsRecursive();
}

void Scene::updateModelMatrices() {
  mRootNode->updateGlobalModelMatrixRecursive();
}

} // namespace scene
} // namespace svulkan2
