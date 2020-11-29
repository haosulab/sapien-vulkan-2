#include "glsl_compiler.h"
#include "svulkan2/common/fs.h"
#include "svulkan2/common/layout.h"

namespace svulkan2 {

class GbufferShaderConfig {
 public:
  void parseGLSL(std::string const &vertFile, std::string const &fragFile); 
};

} // namespace svulkan2
