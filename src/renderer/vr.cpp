#include "svulkan2/renderer/vr.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace renderer {

VRDisplay::VRDisplay() : mSystem(vr::VRSystem()) {
  mContext = core::Context::Get();
  if (!mSystem) {
    throw std::runtime_error("VR is not supported");
  }
  initDevices();
}

void VRDisplay::initDevices() {
  vr::TrackedDeviceIndex_t hmd = vr::k_unTrackedDeviceIndexInvalid;
  for (vr::TrackedDeviceIndex_t di = 0; di < vr::k_unMaxTrackedDeviceCount; ++di) {
    switch (mSystem->GetTrackedDeviceClass(di)) {
    case vr::TrackedDeviceClass::TrackedDeviceClass_HMD:
      hmd = di;
      break;
    default:
      break;
    }
  }
  if (hmd != 0) {
    throw std::runtime_error("Failed to find HMD");
  }
}

std::array<uint32_t, 2> VRDisplay::getScreenSize() const {
  uint32_t width{};
  uint32_t height{};
  mSystem->GetRecommendedRenderTargetSize(&width, &height);
  return {width, height};
}

glm::mat4 VRDisplay::getEyePoseLeft() const {
  vr::HmdMatrix34_t mat = mSystem->GetEyeToHeadTransform(vr::EVREye::Eye_Left);
  return glm::mat4(mat.m[0][0], mat.m[1][0], mat.m[2][0], 0.f, mat.m[0][1], mat.m[1][1],
                   mat.m[2][1], 0.f, mat.m[0][2], mat.m[1][2], mat.m[2][2], 0.f, mat.m[0][3],
                   mat.m[1][3], mat.m[2][3], 1.f);
}

glm::mat4 VRDisplay::getEyePoseRight() const {
  vr::HmdMatrix34_t mat = mSystem->GetEyeToHeadTransform(vr::EVREye::Eye_Right);
  return glm::mat4(mat.m[0][0], mat.m[1][0], mat.m[2][0], 0.f, mat.m[0][1], mat.m[1][1],
                   mat.m[2][1], 0.f, mat.m[0][2], mat.m[1][2], mat.m[2][2], 0.f, mat.m[0][3],
                   mat.m[1][3], mat.m[2][3], 1.f);
}

VRDisplay::Frustum VRDisplay::getCameraFrustumLeft() const {
  VRDisplay::Frustum frustum;
  // NOTE
  // GetProjectionRaw should return left, right, top, bottom
  // But in reality it returns left, right, bottom, top
  // see https://github.com/ValveSoftware/openvr/issues/816
  mSystem->GetProjectionRaw(vr::EVREye::Eye_Left, &frustum.left, &frustum.right, &frustum.bottom,
                            &frustum.top);
  return frustum;
}

VRDisplay::Frustum VRDisplay::getCameraFrustumRight() const {
  VRDisplay::Frustum frustum;
  mSystem->GetProjectionRaw(vr::EVREye::Eye_Right, &frustum.left, &frustum.right, &frustum.bottom,
                            &frustum.top);
  return frustum;
}

std::vector<vr::TrackedDeviceIndex_t> VRDisplay::getControllers() {
  std::vector<vr::TrackedDeviceIndex_t> controllers;
  for (vr::TrackedDeviceIndex_t di = 0; di < vr::k_unMaxTrackedDeviceCount; ++di) {
    switch (mSystem->GetTrackedDeviceClass(di)) {
    case vr::TrackedDeviceClass::TrackedDeviceClass_Controller:
      controllers.push_back(di);
      break;
    default:
      break;
    }
  }
  return controllers;
}

glm::mat4 VRDisplay::getControllerPose(vr::TrackedDeviceIndex_t id) const {
  return mDevicePoses.at(id);
}

glm::mat4 VRDisplay::getHMDPose() const { return mDevicePoses.at(0); }

static glm::mat4 VRPoseToMat4(vr::HmdMatrix34_t const &pose) {
  // clang-format off
  return glm::mat4(pose.m[0][0], pose.m[1][0], pose.m[2][0], 0.0,
                   pose.m[0][1], pose.m[1][1], pose.m[2][1], 0.0,
                   pose.m[0][2], pose.m[1][2], pose.m[2][2], 0.0,
                   pose.m[0][3], pose.m[1][3], pose.m[2][3], 1.0f);
  // clang-format on
}

void VRDisplay::handleInput() {
  // Process SteamVR controller state
  for (vr::TrackedDeviceIndex_t unDevice = 0; unDevice < vr::k_unMaxTrackedDeviceCount;
       unDevice++) {
    vr::VRControllerState_t state;
    if (mSystem->GetControllerState(unDevice, &state, sizeof(state))) {
      mDeviceState[unDevice] = state;
    }
  }
}

uint64_t VRDisplay::getControllerButtonPressed(vr::TrackedDeviceIndex_t id) {
  return mDeviceState.at(id).ulButtonPressed;
}

uint64_t VRDisplay::getControllerButtonTouched(vr::TrackedDeviceIndex_t id) {
  return mDeviceState.at(id).ulButtonTouched;
}

std::array<float, 2> VRDisplay::getControllerAxis(vr::TrackedDeviceIndex_t id, uint32_t axis) {
  if (axis >= vr::k_unControllerStateAxisCount) {
    throw std::runtime_error("requested controller axis out of range");
  }
  return {mDeviceState.at(id).rAxis[axis].x, mDeviceState.at(id).rAxis[axis].y};
}

void VRDisplay::updatePoses() {
  vr::VRCompositor()->WaitGetPoses(mTrackedDevicePose.data(), vr::k_unMaxTrackedDeviceCount, NULL,
                                   0);
  for (int nDevice = 0; nDevice < vr::k_unMaxTrackedDeviceCount; ++nDevice) {
    if (mTrackedDevicePose[nDevice].bPoseIsValid) {
      mDevicePoses[nDevice] = VRPoseToMat4(mTrackedDevicePose[nDevice].mDeviceToAbsoluteTracking);
    }
  }
}

void VRDisplay::renderFrame(core::Image const &left, core::Image const &right) {
  handleInput();

  if (left.getExtent() != right.getExtent()) {
    throw std::runtime_error("left and right images have different dimensions");
  }
  if (left.getExtent() != right.getExtent()) {
    throw std::runtime_error("left and right images have different dimensions");
  }
  if (left.getSampleCount() != vk::SampleCountFlagBits::e1 ||
      right.getSampleCount() != vk::SampleCountFlagBits::e1) {
    throw std::runtime_error("msaa is not supported");
  }
  if (left.getCurrentLayout(0) != vk::ImageLayout::eTransferSrcOptimal ||
      right.getCurrentLayout(0) != vk::ImageLayout::eTransferSrcOptimal) {
    throw std::runtime_error("image layout must be transfer source.");
  }

  vr::VRTextureBounds_t bounds;
  bounds.uMin = 0.0f;
  bounds.uMax = 1.0f;
  bounds.vMin = 0.0f;
  bounds.vMax = 1.0f;

  vr::VRVulkanTextureData_t vulkanData;
  vulkanData.m_nImage = (uint64_t)VkImage(left.getVulkanImage());
  vulkanData.m_pDevice = mContext->getDevice();
  vulkanData.m_pPhysicalDevice = mContext->getPhysicalDevice();
  vulkanData.m_pInstance = mContext->getInstance();
  vulkanData.m_pQueue = mContext->getQueue().getVulkanQueue();
  vulkanData.m_nQueueFamilyIndex = mContext->getGraphicsQueueFamilyIndex();

  vulkanData.m_nWidth = left.getExtent().width;
  vulkanData.m_nHeight = left.getExtent().height;
  vulkanData.m_nFormat = VkFormat(left.getFormat());
  vulkanData.m_nSampleCount = 1;

  vr::Texture_t texture = {&vulkanData, vr::TextureType_Vulkan, vr::ColorSpace_Auto};
  if (vr::VRCompositor()->Submit(vr::Eye_Left, &texture, &bounds) !=
      vr::EVRCompositorError::VRCompositorError_None) {
    throw std::runtime_error("failed to submit");
  }

  vulkanData.m_nImage = (uint64_t)VkImage(right.getVulkanImage());
  if (vr::VRCompositor()->Submit(vr::Eye_Right, &texture, &bounds) !=
      vr::EVRCompositorError::VRCompositorError_None) {
    throw std::runtime_error("failed to submit");
  }
}

} // namespace renderer
} // namespace svulkan2
