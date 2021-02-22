#include "svulkan2/scene/scene.h"
#include <algorithm>

namespace svulkan2 {
namespace scene {

struct PointLightData {
  glm::vec4 position;
  glm::vec4 color;
};

struct DirectionalLightData {
  glm::vec4 direction;
  glm::vec4 color;
};

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
  mObjects.push_back(std::move(obj));
  mObjects.back()->setScene(this);
  mObjects.back()->setParent(parent);
  parent.addChild(*mObjects.back());
  return result;
}

Camera &Scene::addCamera() { return addCamera(getRootNode()); }
Camera &Scene::addCamera(Node &parent) {
  forceRemove();
  auto cam = std::make_unique<Camera>();
  auto &result = *cam;
  mCameras.push_back(std::move(cam));
  mCameras.back()->setScene(this);
  mCameras.back()->setParent(parent);
  parent.addChild(*mCameras.back());
  return result;
}

PointLight &Scene::addPointLight() { return addPointLight(getRootNode()); }
PointLight &Scene::addPointLight(Node &parent) {
  forceRemove();
  auto pointLight = std::make_unique<PointLight>();
  auto &result = *pointLight;
  mPointLights.push_back(std::move(pointLight));
  mPointLights.back()->setScene(this);
  mPointLights.back()->setParent(parent);
  parent.addChild(*mPointLights.back());
  return result;
}

DirectionalLight &Scene::addDirectionalLight() {
  return addDirectionalLight(getRootNode());
}
DirectionalLight &Scene::addDirectionalLight(Node &parent) {
  forceRemove();
  auto directionalLight = std::make_unique<DirectionalLight>();
  auto &result = *directionalLight;
  mDirectionalLights.push_back(std::move(directionalLight));
  mDirectionalLights.back()->setScene(this);
  mDirectionalLights.back()->setParent(parent);
  parent.addChild(*mDirectionalLights.back());
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

void Scene::clearNodes() {
  mNodes.resize(1);
  mObjects.clear();
  mCameras.clear();
  mPointLights.clear();
  mDirectionalLights.clear();
}

void Scene::forceRemove() {
  if (!mRequireForceRemove) {
    return;
  }
  mNodes.erase(std::remove_if(mNodes.begin(), mNodes.end(),
                              [](std::unique_ptr<Node> &node) {
                                return node->isMarkedRemoved();
                              }),
               mNodes.end());
  mObjects.erase(std::remove_if(mObjects.begin(), mObjects.end(),
                                [](std::unique_ptr<Object> &node) {
                                  return node->isMarkedRemoved();
                                }),
                 mObjects.end());
  mCameras.erase(std::remove_if(mCameras.begin(), mCameras.end(),
                                [](std::unique_ptr<Camera> &node) {
                                  return node->isMarkedRemoved();
                                }),
                 mCameras.end());
  mPointLights.erase(std::remove_if(mPointLights.begin(), mPointLights.end(),
                                    [](std::unique_ptr<PointLight> &node) {
                                      return node->isMarkedRemoved();
                                    }),
                     mPointLights.end());
  mDirectionalLights.erase(
      std::remove_if(mDirectionalLights.begin(), mDirectionalLights.end(),
                     [](std::unique_ptr<DirectionalLight> &node) {
                       return node->isMarkedRemoved();
                     }),
      mDirectionalLights.end());

  mRequireForceRemove = false;
}

std::vector<Object *> Scene::getObjects() {
  forceRemove();
  std::vector<Object *> result;
  for (auto &obj : mObjects) {
    result.push_back(obj.get());
  }
  return result;
}

std::vector<PointLight *> Scene::getPointLights() {
  forceRemove();
  std::vector<PointLight *> result;
  for (auto &light : mPointLights) {
    result.push_back(light.get());
  }
  return result;
}

std::vector<DirectionalLight *> Scene::getDirectionalLights() {
  forceRemove();
  std::vector<DirectionalLight *> result;
  for (auto &light : mDirectionalLights) {
    result.push_back(light.get());
  }
  return result;
}

void Scene::updateModelMatrices() {
  mRootNode->updateGlobalModelMatrixRecursive();
}

void Scene::uploadToDevice(core::Buffer &sceneBuffer,
                           StructDataLayout const &sceneLayout) {
  auto pointLights = getPointLights();
  auto directionalLights = getDirectionalLights();
  std::vector<PointLightData> pointLightData;
  std::vector<DirectionalLightData> directionalLightData;
  for (auto light : pointLights) {
    pointLightData.push_back(
        {.position =
             light->getTransform().worldModelMatrix * glm::vec4(0, 0, 0, 1),
         .color = light->getColor()});
  }

  for (auto light : directionalLights) {
    directionalLightData.push_back(
        {.direction =
             light->getTransform().worldModelMatrix * glm::vec4(1, 0, 0, 0),
         .color = light->getColor()});
  }

  sceneBuffer.upload(&mAmbientLight, 16,
                     sceneLayout.elements.at("ambientLight").offset);
  uint32_t numPointLights = mPointLights.size();
  uint32_t numDirectionalLights = mDirectionalLights.size();
  if (sceneLayout.elements.at("pointLights").arrayDim < mPointLights.size()) {
    log::warn("The scene contains more point lights than the maximum number of "
              "point lights in the shader. Truncated.");
    numPointLights = sceneLayout.elements.at("pointLights").arrayDim;
  }
  if (sceneLayout.elements.at("directionalLights").arrayDim <
      mDirectionalLights.size()) {
    log::warn(
        "The scene contains more directional lights than the maximum number of "
        "directional lights in the shader. Truncated.");
    numDirectionalLights =
        sceneLayout.elements.at("directionalLights").arrayDim;
  }
  sceneBuffer.upload(pointLightData.data(),
                     numPointLights * sizeof(PointLightData),
                     sceneLayout.elements.at("pointLights").offset);
  sceneBuffer.upload(directionalLightData.data(),
                     numDirectionalLights * sizeof(PointLightData),
                     sceneLayout.elements.at("directionalLights").offset);
}

} // namespace scene
} // namespace svulkan2
