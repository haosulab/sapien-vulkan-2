#pragma once
#include "layout.h"
#include <memory>

namespace svulkan2 {

/** Renderer options configured by API */
struct RendererConfig {
  std::string shaderDir;
  vk::Format colorFormat1{vk::Format::eR32Sfloat}; // R8Unorm, R32Sfloat
  vk::Format colorFormat4{
      vk::Format::eR32G32B32A32Sfloat}; // R8G8B8A8Unorm, R32G32B32A32Sfloat
  vk::Format depthFormat{vk::Format::eD32Sfloat}; // D32Sfloat
  vk::CullModeFlags culling{vk::CullModeFlagBits::eBack};

  bool operator==(RendererConfig const &other) const = default;
};

} // namespace svulkan2
