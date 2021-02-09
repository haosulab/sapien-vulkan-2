#pragma once
#include "layout.h"
#include <memory>

namespace svulkan2 {

/** Renderer options configured by API */
struct RendererConfig {
  std::string shaderDir;
  vk::Format colorFormat; // R8G8B8A8Unorm, R32G32B32A32Sfloat
  vk::Format depthFormat; // D32Sfloat
  vk::CullModeFlags culling;
  uint32_t maxNumObjects{1000};
};

/** Options configured by the shaders  */
struct ShaderConfig {
  enum MaterialPipeline { eMETALLIC, eSPECULAR, eUNKNOWN } materialPipeline;
  std::shared_ptr<InputDataLayout> vertexLayout;
  std::shared_ptr<StructDataLayout> objectBufferLayout;
  std::shared_ptr<StructDataLayout> sceneBufferLayout;
  std::shared_ptr<StructDataLayout> cameraBufferLayout;
};

} // namespace svulkan2
