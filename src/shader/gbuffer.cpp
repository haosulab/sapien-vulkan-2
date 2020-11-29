#include "svulkan2/shader/gbuffer.h"
#include <spirv_cross/spirv_reflect.hpp>
#include "svulkan2/common/log.h"

namespace svulkan2 {

void GbufferShaderConfig::parseGLSL(std::string const &vertFile,
                                    std::string const &fragFile) {
  GLSLCompiler compiler;
  auto vertSpv = compiler.compileToSpirv(vk::ShaderStageFlagBits::eVertex,
                                         readFile(vertFile));
  log::info("shader compiled: " + vertFile);
  
  auto fragSpv = compiler.compileToSpirv(vk::ShaderStageFlagBits::eFragment,
                                         readFile(fragFile));
  log::info("shader compiled: " + fragFile);
}

} // namespace svulkan2
