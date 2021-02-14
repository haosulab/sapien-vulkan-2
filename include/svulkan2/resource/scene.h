#pragma once
#include "light.h"
#include "svulkan2/common/config.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {

namespace resource {

class SVScene {
  // std::shared_ptr<StructDataLayout> mBufferLayout;
  // std::unique_ptr<core::Buffer> mDeviceBuffer;
  // uint32_t mPointLightCapacity;
  // uint32_t mDirectionalLightCapacity;

  std::vector<PointLight> mPointLights;
  std::vector<DirectionalLight> mDirectionalLights;
  glm::vec4 mAmbientLight{0, 0, 0, 1};

  bool mLightCountChanged{true};
  bool mLightPropertyChanged{true};

public:
  SVScene( // std::shared_ptr<StructDataLayout> bufferLayout
  );

  void addPointLight(PointLight const &pointLight);
  void addDirectionalLight(DirectionalLight const &directionalLight);
  void setPointLightAt(uint32_t index, PointLight const &pointLight);
  void setDirectionalLightAt(uint32_t index,
                             DirectionalLight const &directionalLight);
  void removePointLightAt(uint32_t index);
  void removeDirectionalLightAt(uint32_t index);
  void setAmbeintLight(glm::vec4 const &color);

  inline std::vector<PointLight> const &getPointLights() const {
    return mPointLights;
  }
  inline std::vector<DirectionalLight> const &getDirectionalLights() const {
    return mDirectionalLights;
  }
  inline glm::vec3 getAmbeintLight() const { return mAmbientLight; };

  void uploadToDevice(core::Buffer &sceneBuffer,
                      StructDataLayout const &sceneLayout);
};

} // namespace resource
} // namespace svulkan2
