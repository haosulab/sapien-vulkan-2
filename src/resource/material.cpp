#include "svulkan2/resource/material.h"
#include "svulkan2/core/buffer.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace resource {

void SVMetallicMaterial::setTextures(
    std::shared_ptr<SVTexture> baseColorTexture,
    std::shared_ptr<SVTexture> roughnessTexture,
    std::shared_ptr<SVTexture> normalTexture,
    std::shared_ptr<SVTexture> metallicTexture) {
  mBaseColorTexture = baseColorTexture;
  mRoughnessTexture = roughnessTexture;
  mNormalTexture = normalTexture;
  mMetallicTexture = metallicTexture;

  if (mBaseColorTexture) {
    setBit(mBuffer.textureMask, 0);
  } else {
    unsetBit(mBuffer.textureMask, 0);
  }

  if (mRoughnessTexture) {
    setBit(mBuffer.textureMask, 1);
  } else {
    unsetBit(mBuffer.textureMask, 1);
  }

  if (mNormalTexture) {
    setBit(mBuffer.textureMask, 2);
  } else {
    unsetBit(mBuffer.textureMask, 2);
  }

  if (mMetallicTexture) {
    setBit(mBuffer.textureMask, 3);
  } else {
    unsetBit(mBuffer.textureMask, 3);
  }
}

void SVSpecularMaterial::setTextures(std::shared_ptr<SVTexture> diffuseTexture,
                                     std::shared_ptr<SVTexture> specularTexture,
                                     std::shared_ptr<SVTexture> normalTexture) {
  mDiffuseTexture = diffuseTexture;
  mSpecularTexture = specularTexture;
  mNormalTexture = normalTexture;

  if (diffuseTexture) {
    setBit(mBuffer.textureMask, 0);
  } else {
    unsetBit(mBuffer.textureMask, 0);
  }

  if (specularTexture) {
    setBit(mBuffer.textureMask, 1);
  } else {
    unsetBit(mBuffer.textureMask, 1);
  }

  if (normalTexture) {
    setBit(mBuffer.textureMask, 2);
  } else {
    unsetBit(mBuffer.textureMask, 2);
  }
}

void SVMetallicMaterial::createDeviceResources(core::Context &context) {
  mDeviceBuffer = context.getAllocator().allocateUniformBuffer(
      sizeof(SVMetallicMaterial::Buffer));
}

void SVMetallicMaterial::uploadToDevice() {
  if (!mDeviceBuffer) {
    throw std::runtime_error(
        "failed to upload material buffer: buffer not created");
  }
  mDeviceBuffer->upload(mBuffer);
}

void SVSpecularMaterial::createDeviceResources(core::Context &context) {
  mDeviceBuffer = context.getAllocator().allocateUniformBuffer(
      sizeof(SVSpecularMaterial::Buffer));
}

void SVSpecularMaterial::uploadToDevice() {
  if (!mDeviceBuffer) {
    throw std::runtime_error(
        "failed to upload material buffer: buffer not created");
  }
  mDeviceBuffer->upload(mBuffer);
}

} // namespace resource
} // namespace svulkan2
