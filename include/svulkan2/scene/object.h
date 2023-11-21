#pragma once
#include "node.h"
#include "svulkan2/resource/model.h"
#include "svulkan2/resource/primitive_set.h"
#include <unordered_map>

namespace svulkan2 {

struct CustomData {
  DataType dtype;
  union {
    float floatValue;
    glm::vec2 float2Value;
    glm::vec3 float3Value;
    glm::vec4 float4Value;
    glm::mat4 float44Value;

    int intValue;
    glm::ivec2 int2Value;
    glm::ivec3 int3Value;
    glm::ivec4 int4Value;
  };
};

namespace scene {

class Object : public Node {
  std::shared_ptr<resource::SVModel> mModel;
  glm::uvec4 mSegmentation{0};

  std::unordered_map<std::string, CustomData> mCustomData;
  std::unordered_map<std::string, std::shared_ptr<resource::SVTexture>>
      mCustomTexture;
  std::unordered_map<std::string,
                     std::vector<std::shared_ptr<resource::SVTexture>>>
      mCustomTextureArray;

  int mShadingMode{};
  float mTransparency{};
  bool mCastShadow{true};
  bool mShadeFlat{false};

  vk::CullModeFlagBits mCullMode{vk::CullModeFlagBits::eBack};

public:
  inline Type getType() const override { return Type::eObject; }

  Object(std::shared_ptr<resource::SVModel> model,
         std::string const &name = "");

  void uploadToDevice(core::Buffer &objectBuffer, uint32_t offset,
                      StructDataLayout const &objectLayout);

  void setSegmentation(glm::uvec4 const &segmentation);
  inline glm::uvec4 getSegmentation() const { return mSegmentation; }
  inline std::shared_ptr<resource::SVModel> getModel() const { return mModel; }

  void setCustomDataFloat(std::string const &name, float x);
  void setCustomDataFloat2(std::string const &name, glm::vec2 x);
  void setCustomDataFloat3(std::string const &name, glm::vec3 x);
  void setCustomDataFloat4(std::string const &name, glm::vec4 x);
  void setCustomDataFloat44(std::string const &name, glm::mat4 x);
  void setCustomDataInt(std::string const &name, int x);
  void setCustomDataInt2(std::string const &name, glm::ivec2 x);
  void setCustomDataInt3(std::string const &name, glm::ivec3 x);
  void setCustomDataInt4(std::string const &name, glm::ivec4 x);

  void setCustomTexture(std::string const &name,
                        std::shared_ptr<resource::SVTexture> texture);
  std::shared_ptr<resource::SVTexture> const
  getCustomTexture(std::string const &name) const;

  void setCustomTextureArray(
      std::string const &name,
      std::vector<std::shared_ptr<resource::SVTexture>> textures);
  std::vector<std::shared_ptr<resource::SVTexture>>
  getCustomTextureArray(std::string const &name) const;

  /** used to choose gbuffer pipelines */
  inline void setShadingMode(int mode) { mShadingMode = mode; }
  inline int getShadingMode() const { return mShadingMode; }

  inline void setCullMode(vk::CullModeFlagBits cull) { mCullMode = cull; }
  inline vk::CullModeFlagBits getCullMode() const { return mCullMode; }

  void setTransparency(float transparency);
  inline float getTransparency() const { return mTransparency; }

  void setCastShadow(bool castShadow);
  inline bool getCastShadow() const { return mCastShadow; }

  void setShadeFlat(bool shadeFlat) { mShadeFlat = shadeFlat; };
  inline bool getShadeFlat() const { return mShadeFlat; }

  std::unordered_map<std::string, CustomData> const &getCustomData() const {
    return mCustomData;
  }
};

class LineObject : public Node {
  std::shared_ptr<resource::SVLineSet> mLineSet;
  glm::uvec4 mSegmentation{0};
  // std::unordered_map<std::string, CustomData> mCustomData;
  float mTransparency{};

public:
  inline Type getType() const override { return Type::eObject; }

  LineObject(std::shared_ptr<resource::SVLineSet> lineSet,
             std::string const &name = "");
  inline std::shared_ptr<resource::SVLineSet> getLineSet() const {
    return mLineSet;
  }

  // TODO: remove this function
  void uploadToDevice(core::Buffer &objectBuffer, uint32_t offset,
                      StructDataLayout const &objectLayout);

  void setSegmentation(glm::uvec4 const &segmentation);
  inline glm::uvec4 getSegmentation() const { return mSegmentation; }

  void setTransparency(float transparency);
  inline float getTransparency() const { return mTransparency; }
};

class PointObject : public Node {
  std::shared_ptr<resource::SVPointSet> mPointSet;
  glm::uvec4 mSegmentation{0};
  int mShadingMode{0};
  float mTransparency{0};
  uint32_t mVertexCount{0};

public:
  inline Type getType() const override { return Type::eObject; }

  PointObject(std::shared_ptr<resource::SVPointSet> pointSet,
              std::string const &name = "");
  inline std::shared_ptr<resource::SVPointSet> getPointSet() const {
    return mPointSet;
  }

  /** used to choose pipelines */
  inline void setShadingMode(int mode) { mShadingMode = mode; }
  inline int getShadingMode() const { return mShadingMode; }

  void uploadToDevice(core::Buffer &objectBuffer, uint32_t offset,
                      StructDataLayout const &objectLayout);

  void setSegmentation(glm::uvec4 const &segmentation);
  inline glm::uvec4 getSegmentation() const { return mSegmentation; }

  void setTransparency(float transparency);
  inline float getTransparency() const { return mTransparency; }

  inline uint32_t getMaxVertexCount() const {
    return mPointSet->getVertexCount();
  }

  inline uint32_t getVertexCount() const { return mVertexCount; }
  void setVertexCount(uint32_t count);
};

} // namespace scene
} // namespace svulkan2
