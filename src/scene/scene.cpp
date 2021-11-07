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

struct SpotLightData {
  glm::vec4 position;
  glm::vec4 direction;
  glm::vec4 color;
};

struct LightBufferData {
  glm::mat4 viewMatrix;
  glm::mat4 viewMatrixInverse;
  glm::mat4 projectionMatrix;
  glm::mat4 projectionMatrixInverse;
};

Scene::Scene() {
  mNodes.push_back(std::make_unique<Node>());
  mRootNode = mNodes.back().get();
  mRootNode->setScene(this);
}

Node &Scene::addNode(Node &parent, Transform const &transform) {
  updateVersion();
  forceRemove();
  mNodes.push_back(std::make_unique<Node>());
  mNodes.back()->setScene(this);
  mNodes.back()->setParent(parent);
  parent.addChild(*mNodes.back());

  // update global matrix now, so the next call to update the scene
  // will update the current matrix into the previous matrix
  mNodes.back()->setTransform(transform);
  mNodes.back()->updateGlobalModelMatrixRecursive();
  return *mNodes.back();
}

Node &Scene::addNode(Transform const &transform) {
  return addNode(getRootNode(), transform);
}

Object &Scene::addObject(std::shared_ptr<resource::SVModel> model,
                         Transform const &transform) {
  return addObject(getRootNode(), model, transform);
}

Object &Scene::addObject(Node &parent, std::shared_ptr<resource::SVModel> model,
                         Transform const &transform) {
  updateVersion();
  forceRemove();
  auto obj = std::make_unique<Object>(model);
  auto &result = *obj;
  mObjects.push_back(std::move(obj));
  mObjects.back()->setScene(this);
  mObjects.back()->setParent(parent);
  parent.addChild(*mObjects.back());

  mObjects.back()->setTransform(transform);
  mObjects.back()->updateGlobalModelMatrixRecursive();
  return result;
}

LineObject &Scene::addLineObject(std::shared_ptr<resource::SVLineSet> lineSet,
                                 Transform const &transform) {
  return addLineObject(getRootNode(), lineSet, transform);
}

LineObject &Scene::addLineObject(Node &parent,
                                 std::shared_ptr<resource::SVLineSet> lineSet,
                                 Transform const &transform) {
  updateVersion();
  forceRemove();
  auto obj = std::make_unique<LineObject>(lineSet);
  auto &result = *obj;
  mLineObjects.push_back(std::move(obj));
  mLineObjects.back()->setScene(this);
  mLineObjects.back()->setParent(parent);
  parent.addChild(*mLineObjects.back());

  mLineObjects.back()->setTransform(transform);
  mLineObjects.back()->updateGlobalModelMatrixRecursive();
  return result;
}

PointObject &
Scene::addPointObject(std::shared_ptr<resource::SVPointSet> pointSet,
                      Transform const &transform) {
  return addPointObject(getRootNode(), pointSet, transform);
}

PointObject &
Scene::addPointObject(Node &parent,
                      std::shared_ptr<resource::SVPointSet> pointSet,
                      Transform const &transform) {
  updateVersion();
  forceRemove();
  auto obj = std::make_unique<PointObject>(pointSet);
  auto &result = *obj;
  mPointObjects.push_back(std::move(obj));
  mPointObjects.back()->setScene(this);
  mPointObjects.back()->setParent(parent);
  parent.addChild(*mPointObjects.back());

  mPointObjects.back()->setTransform(transform);
  mPointObjects.back()->updateGlobalModelMatrixRecursive();
  return result;
}

Camera &Scene::addCamera(Transform const &transform) {
  return addCamera(getRootNode(), transform);
}
Camera &Scene::addCamera(Node &parent, Transform const &transform) {
  updateVersion();
  forceRemove();
  auto cam = std::make_unique<Camera>();
  auto &result = *cam;
  mCameras.push_back(std::move(cam));
  mCameras.back()->setScene(this);
  mCameras.back()->setParent(parent);

  mCameras.back()->setTransform(transform);
  mCameras.back()->updateGlobalModelMatrixRecursive();
  parent.addChild(*mCameras.back());
  return result;
}

PointLight &Scene::addPointLight() { return addPointLight(getRootNode()); }
PointLight &Scene::addPointLight(Node &parent) {
  updateVersion();
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
  updateVersion();
  forceRemove();
  auto directionalLight = std::make_unique<DirectionalLight>();
  auto &result = *directionalLight;
  mDirectionalLights.push_back(std::move(directionalLight));
  mDirectionalLights.back()->setScene(this);
  mDirectionalLights.back()->setParent(parent);
  parent.addChild(*mDirectionalLights.back());
  return result;
}

SpotLight &Scene::addSpotLight() { return addSpotLight(getRootNode()); }
SpotLight &Scene::addSpotLight(Node &parent) {
  updateVersion();
  forceRemove();
  auto spotLight = std::make_unique<SpotLight>();
  auto &result = *spotLight;
  mSpotLights.push_back(std::move(spotLight));
  mSpotLights.back()->setScene(this);
  mSpotLights.back()->setParent(parent);
  parent.addChild(*mSpotLights.back());
  return result;
}

TexturedLight &Scene::addTexturedLight() {
  return addTexturedLight(getRootNode());
}
TexturedLight &Scene::addTexturedLight(Node &parent) {
  updateVersion();
  forceRemove();
  auto texturedLight = std::make_unique<TexturedLight>();
  auto &result = *texturedLight;
  mTexturedLights.push_back(std::move(texturedLight));
  mTexturedLights.back()->setScene(this);
  mTexturedLights.back()->setParent(parent);
  parent.addChild(*mTexturedLights.back());
  return result;
}

void Scene::removeNode(Node &node) {
  updateVersion();
  mRequireForceRemove = true;
  node.markRemovedRecursive();
  node.getParent().removeChild(node);
}

void Scene::clearNodes() {
  updateVersion();
  mNodes.resize(1);
  mObjects.clear();
  mLineObjects.clear();
  mPointObjects.clear();
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
  mLineObjects.erase(std::remove_if(mLineObjects.begin(), mLineObjects.end(),
                                    [](std::unique_ptr<LineObject> &node) {
                                      return node->isMarkedRemoved();
                                    }),
                     mLineObjects.end());
  mPointObjects.erase(std::remove_if(mPointObjects.begin(), mPointObjects.end(),
                                     [](std::unique_ptr<PointObject> &node) {
                                       return node->isMarkedRemoved();
                                     }),
                      mPointObjects.end());
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
  mSpotLights.erase(std::remove_if(mSpotLights.begin(), mSpotLights.end(),
                                   [](std::unique_ptr<SpotLight> &node) {
                                     return node->isMarkedRemoved();
                                   }),
                    mSpotLights.end());

  mTexturedLights.erase(
      std::remove_if(mTexturedLights.begin(), mTexturedLights.end(),
                     [](std::unique_ptr<TexturedLight> &node) {
                       return node->isMarkedRemoved();
                     }),
      mTexturedLights.end());

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

std::vector<LineObject *> Scene::getLineObjects() {
  forceRemove();
  std::vector<LineObject *> result;
  for (auto &obj : mLineObjects) {
    result.push_back(obj.get());
  }
  return result;
}

std::vector<PointObject *> Scene::getPointObjects() {
  forceRemove();
  std::vector<PointObject *> result;
  for (auto &obj : mPointObjects) {
    result.push_back(obj.get());
  }
  return result;
}

std::vector<Camera *> Scene::getCameras() {
  forceRemove();
  std::vector<Camera *> result;
  for (auto &obj : mCameras) {
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

std::vector<SpotLight *> Scene::getSpotLights() {
  forceRemove();
  std::vector<SpotLight *> result;
  for (auto &light : mSpotLights) {
    result.push_back(light.get());
  }
  return result;
}

std::vector<TexturedLight *> Scene::getTexturedLights() {
  forceRemove();
  std::vector<TexturedLight *> result;
  for (auto &light : mTexturedLights) {
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
  auto spotLights = getSpotLights();
  auto texturedLights = getTexturedLights();

  std::vector<PointLightData> pointLightData;
  std::vector<DirectionalLightData> directionalLightData;
  std::vector<SpotLightData> spotLightData;
  std::vector<SpotLightData> texturedLightData;
  for (auto light : pointLights) {
    pointLightData.push_back(
        {.position =
             light->getTransform().worldModelMatrix * glm::vec4(0, 0, 0, 1),
         .color = glm::vec4(light->getColor(), 0)});
  }

  for (auto light : directionalLights) {
    directionalLightData.push_back(
        {.direction =
             light->getTransform().worldModelMatrix * glm::vec4(0, 0, -1, 0),
         .color = glm::vec4(light->getColor(), 0)});
  }

  for (auto light : spotLights) {
    spotLightData.push_back(
        {.position = glm::vec4(light->getPosition(), 1),
         .direction = glm::vec4(light->getDirection(), light->getFov()),
         .color = glm::vec4(light->getColor(), light->getFovSmall())});
  }

  for (auto light : texturedLights) {
    texturedLightData.push_back(
        {.position = glm::vec4(light->getPosition(), 1),
         .direction = glm::vec4(light->getDirection(), light->getFov()),
         .color = glm::vec4(light->getColor(), light->getFovSmall())});
  }

  sceneBuffer.upload(&mAmbientLight, 16,
                     sceneLayout.elements.at("ambientLight").offset);
  uint32_t numPointLights = mPointLights.size();
  uint32_t numDirectionalLights = mDirectionalLights.size();
  uint32_t numSpotLights = mSpotLights.size();
  uint32_t numTexturedLights = mTexturedLights.size();
  uint32_t maxNumPointLights =
      sceneLayout.elements.at("pointLights").size /
      sceneLayout.elements.at("pointLights").member->size;
  uint32_t maxNumDirectionalLights =
      sceneLayout.elements.at("directionalLights").size /
      sceneLayout.elements.at("directionalLights").member->size;
  uint32_t maxNumSpotLights =
      sceneLayout.elements.at("spotLights").size /
      sceneLayout.elements.at("spotLights").member->size;
  uint32_t maxNumTexturedLights =
      sceneLayout.elements.at("texturedLights").size /
      sceneLayout.elements.at("texturedLights").member->size;

  if (maxNumPointLights < mPointLights.size()) {
    log::warn("The scene contains more point lights than the maximum number of "
              "point lights in the shader. Truncated.");
    numPointLights = maxNumPointLights;
  }
  if (maxNumDirectionalLights < mDirectionalLights.size()) {
    log::warn(
        "The scene contains more directional lights than the maximum number of "
        "directional lights in the shader. Truncated.");
    numDirectionalLights = maxNumDirectionalLights;
  }
  if (maxNumSpotLights < mSpotLights.size()) {
    log::warn("The scene contains more spot lights than the maximum number of "
              "spot lights in the shader. Truncated.");
    numSpotLights = maxNumSpotLights;
  }
  if (maxNumTexturedLights < mTexturedLights.size()) {
    log::warn("The scene contains more textured lights than the maximum number of "
              "textured lights in the shader. Truncated.");
    numTexturedLights = maxNumTexturedLights;
  }

  sceneBuffer.upload(pointLightData.data(),
                     numPointLights * sizeof(PointLightData),
                     sceneLayout.elements.at("pointLights").offset);
  sceneBuffer.upload(directionalLightData.data(),
                     numDirectionalLights * sizeof(DirectionalLightData),
                     sceneLayout.elements.at("directionalLights").offset);
  sceneBuffer.upload(spotLightData.data(),
                     numSpotLights * sizeof(SpotLightData),
                     sceneLayout.elements.at("spotLights").offset);
  sceneBuffer.upload(texturedLightData.data(),
                     numTexturedLights * sizeof(SpotLightData),
                     sceneLayout.elements.at("texturedLights").offset);
}

void Scene::uploadShadowToDevice(
    core::Buffer &shadowBuffer,
    std::vector<std::unique_ptr<core::Buffer>> &lightBuffers,
    StructDataLayout const &shadowLayout) {

  uint32_t maxNumDirectionalLightShadows =
      shadowLayout.elements.at("directionalLightBuffers").size /
      shadowLayout.elements.at("directionalLightBuffers").member->size;
  uint32_t maxNumPointLightShadows =
      shadowLayout.elements.at("pointLightBuffers").size /
      shadowLayout.elements.at("pointLightBuffers").member->size / 6;
  uint32_t maxNumSpotLightShadows =
      shadowLayout.elements.at("spotLightBuffers").size /
      shadowLayout.elements.at("spotLightBuffers").member->size;
  uint32_t maxNumTexturedLightShadows =
      shadowLayout.elements.at("texturedLightBuffers").size /
      shadowLayout.elements.at("texturedLightBuffers").member->size;

  uint32_t lightBufferIndex = 0;
  {
    std::vector<LightBufferData> directionalLightShadowData;
    uint32_t numDirectionalLightShadows = 0;
    for (auto &l : getDirectionalLights()) {
      if (l->isShadowEnabled()) {
        if (numDirectionalLightShadows >= maxNumDirectionalLightShadows) {
          throw std::runtime_error("The scene contains too many directional "
                                   "lights that cast shadows.");
          break;
        }
        numDirectionalLightShadows++;

        auto modelMat = l->getTransform().worldModelMatrix;
        auto projMat = l->getShadowProjectionMatrix();
        directionalLightShadowData.push_back({
            .viewMatrix = glm::affineInverse(modelMat),
            .viewMatrixInverse = modelMat,
            .projectionMatrix = projMat,
            .projectionMatrixInverse = glm::inverse(projMat),
        });
        lightBuffers[lightBufferIndex++]->upload(
            &directionalLightShadowData.back(), sizeof(LightBufferData));
      } else {
        break;
      }
    }
    shadowBuffer.upload(
        directionalLightShadowData.data(),
        directionalLightShadowData.size() * sizeof(LightBufferData),
        shadowLayout.elements.at("directionalLightBuffers").offset);
  }

  {
    std::vector<LightBufferData> pointLightShadowData;
    uint32_t numPointLightShadows = 0;
    for (auto &l : getPointLights()) {
      if (l->isShadowEnabled()) {
        if (numPointLightShadows >= maxNumPointLightShadows) {
          throw std::runtime_error("The scene contains too many point "
                                   "lights that cast shadows.");
          // log::warn(
          //     "The scene contains too many point lights that cast shadows. "
          //     "Truncated.");
          break;
        }
        numPointLightShadows++;
        auto modelMats = PointLight::getModelMatrices(
            glm::vec3(l->getTransform().worldModelMatrix[3]));
        auto projMat = l->getShadowProjectionMatrix();
        for (uint32_t i = 0; i < 6; ++i) {
          pointLightShadowData.push_back({
              .viewMatrix = glm::affineInverse(modelMats[i]),
              .viewMatrixInverse = modelMats[i],
              .projectionMatrix = projMat,
              .projectionMatrixInverse = glm::inverse(projMat),
          });
          lightBuffers[lightBufferIndex++]->upload(&pointLightShadowData.back(),
                                                   sizeof(LightBufferData));
        }
      } else {
        break;
      }
    }
    shadowBuffer.upload(pointLightShadowData.data(),
                        pointLightShadowData.size() * sizeof(LightBufferData),
                        shadowLayout.elements.at("pointLightBuffers").offset);
  }

  {
    std::vector<LightBufferData> spotLightShadowData;
    uint32_t numSpotLightShadows = 0;
    for (auto &l : getSpotLights()) {
      if (l->isShadowEnabled()) {
        if (numSpotLightShadows >= maxNumSpotLightShadows) {
          throw std::runtime_error("The scene contains too many spot "
                                   "lights that cast shadows.");
          break;
        }
        numSpotLightShadows++;

        auto modelMat = l->getTransform().worldModelMatrix;
        auto projMat = l->getShadowProjectionMatrix();
        spotLightShadowData.push_back({
            .viewMatrix = glm::affineInverse(modelMat),
            .viewMatrixInverse = modelMat,
            .projectionMatrix = projMat,
            .projectionMatrixInverse = glm::inverse(projMat),
        });
        lightBuffers[lightBufferIndex++]->upload(&spotLightShadowData.back(),
                                                 sizeof(LightBufferData));
      } else {
        break;
      }
    }
    shadowBuffer.upload(spotLightShadowData.data(),
                        spotLightShadowData.size() * sizeof(LightBufferData),
                        shadowLayout.elements.at("spotLightBuffers").offset);
  }

  {
    std::vector<LightBufferData> texturedLightShadowData;
    uint32_t numTexturedLightShadows = 0;
    for (auto l : getTexturedLights()) {
      if (numTexturedLightShadows >= maxNumTexturedLightShadows) {
        throw std::runtime_error(
            "The scene contains too many textured lights.");
      }
      numTexturedLightShadows++;
      auto modelMat = l->getTransform().worldModelMatrix;
      auto projMat = l->getShadowProjectionMatrix();

      texturedLightShadowData.push_back({
          .viewMatrix = glm::affineInverse(modelMat),
          .viewMatrixInverse = modelMat,
          .projectionMatrix = projMat,
          .projectionMatrixInverse = glm::inverse(projMat),
      });
      lightBuffers[lightBufferIndex++]->upload(&texturedLightShadowData.back(),
                                               sizeof(LightBufferData));
    }
    shadowBuffer.upload(
        texturedLightShadowData.data(),
        texturedLightShadowData.size() * sizeof(LightBufferData),
        shadowLayout.elements.at("texturedLightBuffers").offset);
  }
}

void Scene::reorderLights() {
  updateVersion();
  std::sort(mPointLights.begin(), mPointLights.end(), [](auto &l1, auto &l2) {
    return l1->isShadowEnabled() && !l2->isShadowEnabled();
  });
  std::sort(mDirectionalLights.begin(), mDirectionalLights.end(),
            [](auto &l1, auto &l2) {
              return l1->isShadowEnabled() && !l2->isShadowEnabled();
            });
  std::sort(mSpotLights.begin(), mSpotLights.end(), [](auto &l1, auto &l2) {
    return l1->isShadowEnabled() && !l2->isShadowEnabled();
  });
}

void Scene::updateVersion() {
  mVersion++;
  log::info("Scene updated");
}

} // namespace scene
} // namespace svulkan2
