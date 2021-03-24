#pragma once
#include "svulkan2/common/glm.h"
#include "svulkan2/core/allocator.h"
#include "texture.h"

namespace svulkan2 {
namespace resource {

inline void setBit(int &number, int bit) { number |= 1 << bit; }
inline void unsetBit(int &number, int bit) { number &= ~(1 << bit); }

class SVMaterial {
protected:
  bool mRequiresBufferUpload{true};
  bool mRequiresTextureUpload{true};
  vk::UniqueDescriptorSet mDescriptorSet;

public:
  inline vk::DescriptorSet getDescriptorSet() const {
    return mDescriptorSet.get();
  }
  virtual void uploadToDevice(core::Context &context) = 0;
  virtual float getOpacity() const = 0;
  virtual ~SVMaterial() = default;
};

class SVMetallicMaterial : public SVMaterial {
  struct Buffer {
    glm::vec4 baseColor;
    float fresnel;
    float roughness;
    float metallic;
    float transparency;
    int textureMask;
  } mBuffer;
  std::shared_ptr<SVTexture> mBaseColorTexture;
  std::shared_ptr<SVTexture> mRoughnessTexture;
  std::shared_ptr<SVTexture> mNormalTexture;
  std::shared_ptr<SVTexture> mMetallicTexture;

  std::unique_ptr<core::Buffer> mDeviceBuffer;

public:
  inline SVMetallicMaterial(glm::vec4 baseColor = {0, 0, 0, 1},
                            float fresnel = 0, float roughness = 1,
                            float metallic = 0, float transparency = 0) {
    mBuffer = {baseColor, fresnel, roughness, metallic, transparency, 0};
  }

  void setBaseColor(glm::vec4 baseColor);
  glm::vec4 getBaseColor() const;

  void setRoughness(float roughness);
  float getRoughness() const;

  void setFresnel(float fresnel);
  float getFresnel() const;

  void setMetallic(float metallic);
  float getMetallic() const;

  void setTextures(std::shared_ptr<SVTexture> baseColorTexture,
                   std::shared_ptr<SVTexture> roughnessTexture,
                   std::shared_ptr<SVTexture> normalTexture,
                   std::shared_ptr<SVTexture> metallicTexture);

  virtual void uploadToDevice(core::Context &context) override;
  inline float getOpacity() const override { return mBuffer.baseColor.a; }
};

class SVSpecularMaterial : public SVMaterial {
  struct Buffer {
    glm::vec4 diffuse;
    glm::vec4 specular;
    float transparency;
    int textureMask;
  } mBuffer;
  std::shared_ptr<SVTexture> mDiffuseTexture;
  std::shared_ptr<SVTexture> mSpecularTexture;
  std::shared_ptr<SVTexture> mNormalTexture;

  std::unique_ptr<core::Buffer> mDeviceBuffer;

public:
  inline SVSpecularMaterial(glm::vec4 diffuse = {0, 0, 0, 1},
                            glm::vec4 specular = {0, 0, 0, 0},
                            float transparency = 0) {
    mBuffer = {diffuse, specular, transparency, 0};
  }

  void setTextures(std::shared_ptr<SVTexture> diffuseTexture,
                   std::shared_ptr<SVTexture> specularTexture,
                   std::shared_ptr<SVTexture> normalTexture);

  virtual void uploadToDevice(core::Context &context) override;
  inline float getOpacity() const override { return mBuffer.diffuse.a; }
};

} // namespace resource
} // namespace svulkan2
