#pragma once
#include "svulkan2/common/glm.h"
#include <memory>
#include <openvr/openvr.h>
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

  std::vector<vr::TrackedDeviceIndex_t> getControllers();
  glm::mat4 getControllerPose(vr::TrackedDeviceIndex_t id) const;
  glm::mat4 getHMDPose() const;

  uint64_t getControllerButtonPressed(vr::TrackedDeviceIndex_t id);
  uint64_t getControllerButtonTouched(vr::TrackedDeviceIndex_t id);
  std::array<float, 2> getControllerAxis(vr::TrackedDeviceIndex_t id, uint32_t axis);

  void handleInput();
  void updatePoses();
  void renderFrame(core::Image const &left, core::Image const &right);

  ~VRDisplay() {}

private:
  std::shared_ptr<core::Context> mContext;
  vr::IVRSystem *mSystem;

  std::array<vr::TrackedDevicePose_t, vr::k_unMaxTrackedDeviceCount> mTrackedDevicePose{};
  std::array<glm::mat4, vr::k_unMaxTrackedDeviceCount> mDevicePoses{};
  std::array<vr::VRControllerState_t, vr::k_unMaxTrackedDeviceCount> mDeviceState{};
};

} // namespace renderer
} // namespace svulkan2
