#include "layout.h"
#include <memory>

namespace svulkan2 {

/** Renderer options configured by API */
struct RendererConfig {};

/** Options configured by the shaders  */
struct ShaderConfig {
  enum MaterialPipeline { eMETALLIC, eSPECULAR } materialPipeline;
  std::shared_ptr<InputDataLayout> vertexLayout;
};

} // namespace svulkan2
