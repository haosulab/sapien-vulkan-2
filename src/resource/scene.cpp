#include "svulkan2/resource/scene.h"
#include "svulkan2/common/log.h"
#include "svulkan2/core/buffer.h"

namespace svulkan2 {
namespace scene {
class Scene;
}
namespace resource {

SVScene::SVScene() {}

void SVScene::addPointLight(PointLight const &pointLight) {
  mLightPropertyChanged = true;
  mLightCountChanged = true;
  mPointLights.push_back(pointLight);
}

void SVScene::addDirectionalLight(DirectionalLight const &directionalLight) {
  mLightPropertyChanged = true;
  mLightCountChanged = true;
  mDirectionalLights.push_back(directionalLight);
}

void SVScene::setPointLightAt(uint32_t index, PointLight const &pointLight) {
  mLightPropertyChanged = true;
  if (index < mPointLights.size()) {
    mPointLights[index] = pointLight;
  }
}
void SVScene::setDirectionalLightAt(uint32_t index,
                                    DirectionalLight const &directionalLight) {
  mLightPropertyChanged = true;
  if (index < mDirectionalLights.size()) {
    mDirectionalLights[index] = directionalLight;
  }
}

void SVScene::removePointLightAt(uint32_t index) {
  mLightPropertyChanged = true;
  mLightCountChanged = true;
  if (index < mPointLights.size()) {
    mPointLights.erase(mPointLights.begin() + index);
  }
}

void SVScene::removeDirectionalLightAt(uint32_t index) {
  mLightPropertyChanged = true;
  mLightCountChanged = true;
  if (index < mDirectionalLights.size()) {
    mDirectionalLights.erase(mDirectionalLights.begin() + index);
  }
}

void SVScene::setAmbeintLight(glm::vec4 const &color) {
  mLightPropertyChanged = true;
  mAmbientLight = color;
};

void SVScene::uploadToDevice(core::Buffer &sceneBuffer,
                             StructDataLayout const &sceneLayout) {
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

  sceneBuffer.upload(mPointLights.data(), numPointLights * sizeof(PointLight),
                     sceneLayout.elements.at("pointLights").offset);
  sceneBuffer.upload(mDirectionalLights.data(),
                     numDirectionalLights * sizeof(PointLight),
                     sceneLayout.elements.at("directionalLights").offset);
  mLightPropertyChanged = false;
}

} // namespace resource
} // namespace svulkan2
