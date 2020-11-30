#include "glsl_compiler.h"
#include "reflect.h"
#include "svulkan2/common/fs.h"
#include "svulkan2/common/layout.h"

namespace svulkan2 {

class GbufferShaderConfig {
  std::vector<uint32_t> mVertSpv;
  std::vector<uint32_t> mFragSpv;

  DataLayout mVertexLayout{};
  DataLayout mCameraLayout{};
  DataLayout mObjectLayout{};
  DataLayout mOutputLayout{};

  enum { eSPECULAR, eMETALLIC } mMaterialType;

public:
  void parseGLSL(std::string const &vertFile, std::string const &fragFile);

private:
  void reflectSPV();

  void processVertexInput(spv_reflect::ShaderModule &module);
  void processCamera(spv_reflect::ShaderModule &vertModule);
  void processObject(spv_reflect::ShaderModule &vertModule);
  void processMaterial(spv_reflect::ShaderModule &fragModule);
  void processOutput(spv_reflect::ShaderModule &fragModule);
};

} // namespace svulkan2
