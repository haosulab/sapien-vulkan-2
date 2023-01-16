#include "svulkan2/resource/material.h"
#include "svulkan2/core/buffer.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace resource {

#ifdef TRACK_ALLOCATION
static uint64_t gMaterialId = 1;
static uint64_t gMaterialCount = 0;
#endif

static void updateDescriptorSets(
    vk::Device device, vk::DescriptorSet descriptorSet,
    std::vector<std::tuple<vk::DescriptorType, vk::Buffer,
                           vk::BufferView>> const &bufferData,
    std::vector<std::tuple<vk::ImageView, vk::Sampler>> textureData,
    uint32_t bindingOffset) {

  std::vector<vk::DescriptorBufferInfo> bufferInfos;
  bufferInfos.reserve(bufferData.size());

  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  writeDescriptorSets.reserve(bufferData.size() + textureData.size() ? 1 : 0);

  uint32_t dstBinding = bindingOffset;
  for (auto const &bd : bufferData) {
    bufferInfos.push_back(
        vk::DescriptorBufferInfo(std::get<1>(bd), 0, VK_WHOLE_SIZE));
    writeDescriptorSets.push_back(vk::WriteDescriptorSet(
        descriptorSet, dstBinding++, 0, 1, std::get<0>(bd), nullptr,
        &bufferInfos.back(), std::get<2>(bd) ? &std::get<2>(bd) : nullptr));
  }

  std::vector<vk::DescriptorImageInfo> imageInfos;
  for (auto const &tex : textureData) {
    imageInfos.push_back(
        vk::DescriptorImageInfo(std::get<1>(tex), std::get<0>(tex),
                                vk::ImageLayout::eShaderReadOnlyOptimal));
  }
  if (imageInfos.size()) {
    writeDescriptorSets.push_back(vk::WriteDescriptorSet(
        descriptorSet, dstBinding, 0, imageInfos.size(),
        vk::DescriptorType::eCombinedImageSampler, imageInfos.data()));
  }
  device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

SVMetallicMaterial::SVMetallicMaterial(glm::vec4 emission, glm::vec4 baseColor,
                                       float fresnel, float roughness,
                                       float metallic, float transparency,
                                       float ior, float transmissionRoughness) {
  mBuffer = {emission,     baseColor, fresnel, roughness, metallic,
             transparency, ior,       transmissionRoughness,     0};
#ifdef TRACK_ALLOCATION
  mMaterialId = gMaterialId++;
  log::info("Create Material {}; Total {}", mMaterialId, ++gMaterialCount);
#endif
}

SVMetallicMaterial::~SVMetallicMaterial() {
#ifdef TRACK_ALLOCATION
  log::info("Destroy Material {}, Total {}", mMaterialId, --gMaterialCount);
#endif
}

void SVMetallicMaterial::setEmission(glm::vec4 emission) {
  mRequiresBufferUpload = true;
  mBuffer.emission = emission;
}

glm::vec4 SVMetallicMaterial::getEmission() const { return mBuffer.emission; }

void SVMetallicMaterial::setBaseColor(glm::vec4 baseColor) {
  mRequiresBufferUpload = true;
  mBuffer.baseColor = baseColor;
}

glm::vec4 SVMetallicMaterial::getBaseColor() const { return mBuffer.baseColor; }

void SVMetallicMaterial::setRoughness(float roughness) {
  mRequiresBufferUpload = true;
  mBuffer.roughness = roughness;
}

float SVMetallicMaterial::getRoughness() const { return mBuffer.roughness; }

void SVMetallicMaterial::setFresnel(float fresnel) {
  mRequiresBufferUpload = true;
  mBuffer.fresnel = fresnel;
}

float SVMetallicMaterial::getFresnel() const { return mBuffer.fresnel; }

void SVMetallicMaterial::setMetallic(float metallic) {
  mRequiresBufferUpload = true;
  mBuffer.metallic = metallic;
}

float SVMetallicMaterial::getMetallic() const { return mBuffer.metallic; }

void SVMetallicMaterial::setTransmission(float transmission) {
  mRequiresBufferUpload = true;
  mBuffer.transmission = transmission;
}
float SVMetallicMaterial::getTransmission() const {
  return mBuffer.transmission;
}

void SVMetallicMaterial::setIor(float ior) {
  mRequiresBufferUpload = true;
  mBuffer.ior = ior;
}
float SVMetallicMaterial::getIor() const { return mBuffer.ior; }

void SVMetallicMaterial::setTransmissionRoughness(float roughness) {
  mRequiresBufferUpload = true;
  mBuffer.transmissionRoughness = roughness;
}
float SVMetallicMaterial::getTransmissionRoughness() const {
  return mBuffer.transmissionRoughness;
}

std::shared_ptr<SVTexture> SVMetallicMaterial::getDiffuseTexture() const {
  if ((mBuffer.textureMask & 1) == 0) {
    return nullptr;
  }
  return mBaseColorTexture;
}

std::shared_ptr<SVTexture> SVMetallicMaterial::getRoughnessTexture() const {
  if ((mBuffer.textureMask & 2) == 0) {
    return nullptr;
  }
  return mRoughnessTexture;
}

std::shared_ptr<SVTexture> SVMetallicMaterial::getNormalTexture() const {
  if ((mBuffer.textureMask & 4) == 0) {
    return nullptr;
  }
  return mNormalTexture;
}

std::shared_ptr<SVTexture> SVMetallicMaterial::getMetallicTexture() const {
  if ((mBuffer.textureMask & 8) == 0) {
    return nullptr;
  }
  return mMetallicTexture;
}

std::shared_ptr<SVTexture> SVMetallicMaterial::getEmissionTexture() const {
  if ((mBuffer.textureMask & 16) == 0) {
    return nullptr;
  }
  return mEmissionTexture;
}

void SVMetallicMaterial::setDiffuseTexture(std::shared_ptr<SVTexture> texture) {
  mRequiresTextureUpload = true;
  mBaseColorTexture = texture;
  if (mBaseColorTexture) {
    setBit(mBuffer.textureMask, 0);
  } else {
    unsetBit(mBuffer.textureMask, 0);
  }
}
void SVMetallicMaterial::setRoughnessTexture(
    std::shared_ptr<SVTexture> texture) {
  mRequiresTextureUpload = true;
  mRoughnessTexture = texture;
  if (mRoughnessTexture) {
    setBit(mBuffer.textureMask, 1);
  } else {
    unsetBit(mBuffer.textureMask, 1);
  }
}

void SVMetallicMaterial::setNormalTexture(std::shared_ptr<SVTexture> texture) {
  mRequiresTextureUpload = true;
  mNormalTexture = texture;
  if (mNormalTexture) {
    setBit(mBuffer.textureMask, 2);
  } else {
    unsetBit(mBuffer.textureMask, 2);
  }
}

void SVMetallicMaterial::setMetallicTexture(
    std::shared_ptr<SVTexture> texture) {
  mRequiresTextureUpload = true;
  mMetallicTexture = texture;
  if (mMetallicTexture) {
    setBit(mBuffer.textureMask, 3);
  } else {
    unsetBit(mBuffer.textureMask, 3);
  }
}

void SVMetallicMaterial::setEmissionTexture(
    std::shared_ptr<SVTexture> texture) {
  mRequiresTextureUpload = true;
  mEmissionTexture = texture;
  if (mMetallicTexture) {
    setBit(mBuffer.textureMask, 4);
  } else {
    unsetBit(mBuffer.textureMask, 4);
  }
}

void SVMetallicMaterial::setTextures(
    std::shared_ptr<SVTexture> baseColorTexture,
    std::shared_ptr<SVTexture> roughnessTexture,
    std::shared_ptr<SVTexture> normalTexture,
    std::shared_ptr<SVTexture> metallicTexture,
    std::shared_ptr<SVTexture> emissionTexture) {
  mRequiresTextureUpload = true;

  mBaseColorTexture = baseColorTexture;
  mRoughnessTexture = roughnessTexture;
  mNormalTexture = normalTexture;
  mMetallicTexture = metallicTexture;
  mEmissionTexture = emissionTexture;

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

  if (mEmissionTexture) {
    setBit(mBuffer.textureMask, 4);
  } else {
    unsetBit(mBuffer.textureMask, 4);
  }
}

void SVMetallicMaterial::uploadToDevice() {
  mContext = core::Context::Get();
  std::scoped_lock lock(mContext->getGlobalLock());
  if (!mDeviceBuffer) {
    vk::BufferUsageFlags flags = vk::BufferUsageFlagBits::eUniformBuffer;
    if (mContext->isRayTracingAvailable()) {
      flags |= vk::BufferUsageFlagBits::eStorageBuffer;
    }

    mDeviceBuffer = std::make_unique<core::Buffer>(
        sizeof(SVMetallicMaterial::Buffer), flags, VMA_MEMORY_USAGE_CPU_TO_GPU);

    auto layout = mContext->getMetallicDescriptorSetLayout();
    mDescriptorSet = std::move(
        mContext->getDevice()
            .allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(
                mContext->getDescriptorPool(), 1, &layout))
            .front());
  }

  if (mRequiresBufferUpload) {
    mDeviceBuffer->upload(mBuffer);
    mRequiresBufferUpload = false;
  }

  if (mRequiresTextureUpload) {
    auto defaultTexture = mContext->getResourceManager()->getDefaultTexture();
    std::vector<std::tuple<vk::ImageView, vk::Sampler>> textures;
    if (mBaseColorTexture) {
      mBaseColorTexture->uploadToDevice();
      textures.push_back(
          {mBaseColorTexture->getImageView(), mBaseColorTexture->getSampler()});
    } else {
      defaultTexture->uploadToDevice();
      textures.push_back(
          {defaultTexture->getImageView(), defaultTexture->getSampler()});
    }
    if (mRoughnessTexture) {
      mRoughnessTexture->uploadToDevice();
      textures.push_back(
          {mRoughnessTexture->getImageView(), mRoughnessTexture->getSampler()});
    } else {
      defaultTexture->uploadToDevice();
      textures.push_back(
          {defaultTexture->getImageView(), defaultTexture->getSampler()});
    }
    if (mNormalTexture) {
      mNormalTexture->uploadToDevice();
      textures.push_back(
          {mNormalTexture->getImageView(), mNormalTexture->getSampler()});
    } else {
      defaultTexture->uploadToDevice();
      textures.push_back(
          {defaultTexture->getImageView(), defaultTexture->getSampler()});
    }
    if (mMetallicTexture) {
      mMetallicTexture->uploadToDevice();
      textures.push_back(
          {mMetallicTexture->getImageView(), mMetallicTexture->getSampler()});
    } else {
      defaultTexture->uploadToDevice();
      textures.push_back(
          {defaultTexture->getImageView(), defaultTexture->getSampler()});
    }
    if (mEmissionTexture) {
      mEmissionTexture->uploadToDevice();
      textures.push_back(
          {mEmissionTexture->getImageView(), mEmissionTexture->getSampler()});
    } else {
      defaultTexture->uploadToDevice();
      textures.push_back(
          {defaultTexture->getImageView(), defaultTexture->getSampler()});
    }
    updateDescriptorSets(mContext->getDevice(), mDescriptorSet.get(),
                         {{vk::DescriptorType::eUniformBuffer,
                           mDeviceBuffer->getVulkanBuffer(), nullptr}},
                         textures, 0);
    mRequiresTextureUpload = false;
  }
}

void SVMetallicMaterial::removeFromDevice() {
  if (mContext) {
    std::scoped_lock lock(mContext->getGlobalLock());
  }
  mRequiresBufferUpload = true;
  if (mBaseColorTexture) {
    mBaseColorTexture->removeFromDevice();
  }
  if (mRoughnessTexture) {
    mRoughnessTexture->removeFromDevice();
  }
  if (mNormalTexture) {
    mNormalTexture->removeFromDevice();
  }
  if (mMetallicTexture) {
    mMetallicTexture->removeFromDevice();
  }
  if (mEmissionTexture) {
    mEmissionTexture->removeFromDevice();
  }
  mDescriptorSet.reset();
  mDeviceBuffer.reset();
}

core::Buffer &SVMetallicMaterial::getDeviceBuffer() const {
  if (!mDeviceBuffer) {
    throw std::runtime_error("failed to get device buffer: not uploaded");
  }
  return *mDeviceBuffer;
}

} // namespace resource
} // namespace svulkan2
