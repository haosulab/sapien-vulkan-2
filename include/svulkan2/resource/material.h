#pragma once
#include "svulkan2/common/glm.h"
#include "svulkan2/core/allocator.h"
#include "texture.h"

namespace svulkan2 {
namespace resource {

inline void setBit(int &number, int bit) { number |= 1 << bit; }
inline void unsetBit(int &number, int bit) { number &= ~(1 << bit); }
inline int getBit(int number, int bit) { return (number >> bit) & 1; }

class SVMaterial {
public:
  inline vk::DescriptorSet getDescriptorSet() const {
    return mDescriptorSet.get();
  }
  virtual void uploadToDevice() = 0;
  virtual void removeFromDevice() = 0;
  virtual float getOpacity() const = 0;

  inline int getGlobalIndex() const { return mGlobalIndex; }
  inline void setGlobalIndex(int index) { mGlobalIndex = index; }

  virtual ~SVMaterial() = default;

protected:
  std::shared_ptr<core::Context> mContext; // keep alive for descriptor set
  bool mRequiresBufferUpload{true};
  bool mRequiresTextureUpload{true};

  vk::UniqueDescriptorSet mDescriptorSet;

  int mGlobalIndex{-1}; // global index global array
};

class SVMetallicMaterial : public SVMaterial {

  enum TextureBit {
    eBaseColor = 0,
    eRoughness = 1,
    eNormal = 2,
    eMetallic = 3,
    eEmission = 4,
    eTransmission = 5
  };

  struct Buffer {
    glm::vec4 emission{};
    glm::vec4 baseColor{};
    float fresnel{};
    float roughness{};
    float metallic{};
    float transmission{};
    float ior{};
    float transmissionRoughness{};
    int textureMask{};
    int padding1;
  } mBuffer{};
  std::shared_ptr<SVTexture> mBaseColorTexture;
  std::shared_ptr<SVTexture> mRoughnessTexture;
  std::shared_ptr<SVTexture> mNormalTexture;
  std::shared_ptr<SVTexture> mMetallicTexture;
  std::shared_ptr<SVTexture> mEmissionTexture;
  std::shared_ptr<SVTexture> mTransmissionTexture;

  std::unique_ptr<core::Buffer> mDeviceBuffer;

#ifdef TRACK_ALLOCATION
  uint64_t mMaterialId{};
#endif

public:
  SVMetallicMaterial(glm::vec4 emission = {0, 0, 0, 1},
                     glm::vec4 baseColor = {0, 0, 0, 1}, float fresnel = 0,
                     float roughness = 1, float metallic = 0,
                     float transparency = 0, float ior = 1.f,
                     float transmissionRoughness = 0.f);

  void setEmission(glm::vec4 emission);
  glm::vec4 getEmission() const;

  void setBaseColor(glm::vec4 baseColor);
  glm::vec4 getBaseColor() const;

  void setRoughness(float roughness);
  float getRoughness() const;

  void setFresnel(float fresnel);
  float getFresnel() const;

  void setMetallic(float metallic);
  float getMetallic() const;

  void setTransmission(float transmission);
  float getTransmission() const;

  void setIor(float ior);
  float getIor() const;

  void setTransmissionRoughness(float roughness);
  float getTransmissionRoughness() const;

  void setDiffuseTexture(std::shared_ptr<SVTexture> texture);
  void setRoughnessTexture(std::shared_ptr<SVTexture> texture);
  void setNormalTexture(std::shared_ptr<SVTexture> texture);
  void setMetallicTexture(std::shared_ptr<SVTexture> texture);
  void setEmissionTexture(std::shared_ptr<SVTexture> texture);
  void setTransmissionTexture(std::shared_ptr<SVTexture> texture);

  std::shared_ptr<SVTexture> getDiffuseTexture() const;
  std::shared_ptr<SVTexture> getRoughnessTexture() const;
  std::shared_ptr<SVTexture> getNormalTexture() const;
  std::shared_ptr<SVTexture> getMetallicTexture() const;
  std::shared_ptr<SVTexture> getEmissionTexture() const;
  std::shared_ptr<SVTexture> getTransmissionTexture() const;

  void setTextures(std::shared_ptr<SVTexture> baseColorTexture,
                   std::shared_ptr<SVTexture> roughnessTexture,
                   std::shared_ptr<SVTexture> normalTexture,
                   std::shared_ptr<SVTexture> metallicTexture,
                   std::shared_ptr<SVTexture> emissionTexture,
                   std::shared_ptr<SVTexture> transmissionTexture);

  void uploadToDevice() override;
  void removeFromDevice() override;
  inline float getOpacity() const override { return mBuffer.baseColor.a; }

  core::Buffer &getDeviceBuffer() const;

  virtual ~SVMetallicMaterial();
};

} // namespace resource
} // namespace svulkan2
