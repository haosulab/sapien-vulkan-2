#pragma once
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace svulkan2 {

class GLSLCompiler {
public:
  std::vector<std::uint32_t> compileToSpirv(vk::ShaderStageFlagBits shaderStage,
                                            std::vector<char> const &glslCode);
};

} // namespace svulkan2
