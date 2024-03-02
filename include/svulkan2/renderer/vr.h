#pragma once
#include "svulkan2/common/glm.h"
#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace svulkan2 {
namespace core {
class Context;
class Image;
} // namespace core
namespace renderer {

class VRDisplay {
public:
  VRDisplay();
  void initDevices();

  std::array<uint32_t, 2> getScreenSize() const;
  glm::mat4 getEyePoseLeft() const;
  glm::mat4 getEyePoseRight() const;

  struct Frustum {
    float left;
    float right;
    float top;
    float bottom;
  };
  Frustum getCameraFrustumLeft() const;
  Frustum getCameraFrustumRight() const;

  std::vector<uint32_t> getControllers();
  glm::mat4 getControllerPose(uint32_t id) const;
  glm::mat4 getHMDPose() const;

  uint64_t getControllerButtonPressed(uint32_t id);
  uint64_t getControllerButtonTouched(uint32_t id);
  std::array<float, 2> getControllerAxis(uint32_t id, uint32_t axis);

  void handleInput();
  void updatePoses();
  void renderFrame(core::Image const &left, core::Image const &right);

  ~VRDisplay() {}

private:
  std::shared_ptr<core::Context> mContext;
};

} // namespace renderer
} // namespace svulkan2
