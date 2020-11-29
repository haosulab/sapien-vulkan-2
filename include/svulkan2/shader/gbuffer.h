#include "glsl_compiler.h"
#include "svulkan2/common/fs.h"
#include "svulkan2/common/layout.h"
#include "reflect.h"

namespace svulkan2 {

class GbufferShaderConfig {
  std::vector<uint32_t> mVertSpv;
  std::vector<uint32_t> mFragSpv;

public:
  void parseGLSL(std::string const &vertFile, std::string const &fragFile);

private:
  void reflectSPV();

  void processVertexInput(spv_reflect::ShaderModule &module);
};

} // namespace svulkan2
