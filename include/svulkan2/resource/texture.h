#pragma once
#include <memory>
#include <vector>

namespace svulkan2 {

class VulkanTexture;

class Texture {
  std::unique_ptr<VulkanTexture> mVulkanTexture;
};

} // namespace svulkan2
