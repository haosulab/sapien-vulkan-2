#include "svulkan2/resource/scene.h"
#include "svulkan2/common/log.h"
#include "svulkan2/core/buffer.h"

namespace svulkan2 {
namespace scene {
class Scene;
}
namespace resource {

SVScene::SVScene(std::shared_ptr<StructDataLayout> bufferLayout)
    : mBufferLayout(bufferLayout) {
  mPointLightCapacity = bufferLayout->elements.at("pointLights").arrayDim;
  mDirectionalLightCapacity =
      bufferLayout->elements.at("directionalLights").arrayDim;
}

void SVScene::addPointLight(PointLight const &pointLight) {
  mDirty = true;
  mRequiresRebuld = true;
  if (mPointLights.size() == mPointLightCapacity) {
    log::error(
        "failed to add point light: exceeding shader-specified capacity");
  }
  mPointLights.push_back(pointLight);
}

void SVScene::addDirectionalLight(DirectionalLight const &directionalLight) {
  mDirty = true;
  mRequiresRebuld = true;
  if (mDirectionalLights.size() == mDirectionalLightCapacity) {
    log::error(
        "failed to add directional light: exceeding shader-specified capacity");
  }
  mDirectionalLights.push_back(directionalLight);
}

void SVScene::setPointLightAt(uint32_t index, PointLight const &pointLight) {
  mDirty = true;
  if (index < mPointLights.size()) {
    mPointLights[index] = pointLight;
  }
}
void SVScene::setDirectionalLightAt(uint32_t index,
                                    DirectionalLight const &directionalLight) {
  mDirty = true;
  if (index < mDirectionalLights.size()) {
    mDirectionalLights[index] = directionalLight;
  }
}

void SVScene::removePointLightAt(uint32_t index) {
  mDirty = true;
  mRequiresRebuld = true;
  if (index < mPointLights.size()) {
    mPointLights.erase(mPointLights.begin() + index);
  }
}

void SVScene::removeDirectionalLightAt(uint32_t index) {
  mDirty = true;
  mRequiresRebuld = true;
  if (index < mDirectionalLights.size()) {
    mDirectionalLights.erase(mDirectionalLights.begin() + index);
  }
}

void SVScene::createDeviceResources(core::Context &context) {
  mDeviceBuffer =
      context.getAllocator().allocateUniformBuffer(mBufferLayout->size);
}

void SVScene::setAmbeintLight(glm::vec4 const &color) {
  mDirty = true;
  mAmbientLight = color;
};

void SVScene::uploadToDevice() {
  if (!mDeviceBuffer) {
    throw std::runtime_error(
        "failed to upload scene to device: buffer not created");
  }
  if (mDirty) {
    mDeviceBuffer->upload(&mAmbientLight, 16,
                          mBufferLayout->elements.at("ambientLight").offset);
    mDeviceBuffer->upload(mPointLights.data(),
                          mPointLights.size() * sizeof(PointLight),
                          mBufferLayout->elements.at("pointLights").offset);
    mDeviceBuffer->upload(
        mDirectionalLights.data(),
        mDirectionalLights.size() * sizeof(PointLight),
        mBufferLayout->elements.at("directionalLights").offset);
    // TODO: upload shadow matrix and other stuff
    mDirty = false;
  }
}

} // namespace resource
} // namespace svulkan2
