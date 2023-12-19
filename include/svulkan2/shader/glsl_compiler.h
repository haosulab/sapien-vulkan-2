#pragma once
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace svulkan2 {
namespace fs = std::filesystem;
class GLSLCompiler {
public:
  static std::vector<std::uint32_t> const &
  compileGlslFileCached(vk::ShaderStageFlagBits stage,
                        fs::path const &filepath);

  static std::tuple<std::string, std::vector<std::tuple<std::string, int>>>
  loadGlslCodeWithDebugInfo(fs::path const &filepath);
  // static std::string loadGlslCode(fs::path const &filepath);
  static std::vector<std::uint32_t>
  compileToSpirv(vk::ShaderStageFlagBits shaderStage,
                 std::string const &glslCode,
                 std::vector<std::tuple<std::string, int>> const &debugInfo = {});
  static void InitializeProcess();
  static void FinalizeProcess();

private:
};

} // namespace svulkan2
