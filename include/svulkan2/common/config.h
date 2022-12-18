#pragma once
#include "layout.h"
#include <memory>
#include <unordered_map>

namespace svulkan2 {

/** Renderer options configured by API */
struct RendererConfig {
  std::string shaderDir;

  // default 1-channel texture format
  vk::Format colorFormat1{vk::Format::eR32Sfloat}; // R8Unorm, R32Sfloat

  // default 4-channel texture format
  vk::Format colorFormat4{
      vk::Format::eR32G32B32A32Sfloat}; // R8G8B8A8Unorm, R32G32B32A32Sfloat

  // texture foramt for specific textures
  std::unordered_map<std::string, vk::Format> textureFormat;

  vk::Format depthFormat{vk::Format::eD32Sfloat}; // D32Sfloat
  vk::CullModeFlags culling{vk::CullModeFlagBits::eBack};

  bool operator==(RendererConfig const &other) const = default;
};

} // namespace svulkan2
