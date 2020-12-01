#include "svulkan2/shader/deferred.h"

namespace svulkan2 {
void DeferredPassParser::processInput(spirv_cross::Compiler &compiler) {}
void DeferredPassParser::processOutput(spirv_cross::Compiler &compiler) {}

void DeferredPassParser::processCamera(spirv_cross::Compiler &compiler) {
  mCameraLayout = parseCamera(compiler, 0, 1, "deferred.frag");
}

void DeferredPassParser::processScene(spirv_cross::Compiler &compiler) {
  // uint32_t count;
  // fragModule.EnumeratePushConstants(&count, nullptr);
  // std::vector<SpvReflectInterfaceVariable *> inputVariables(count);
  // fragModule.EnumerateInputVariables(&count, inputVariables.data());
}

void DeferredPassParser::reflectSPV() {
  spirv_cross::Compiler vertComp(mVertSPVCode);
  spirv_cross::Compiler fragComp(mFragSPVCode);
  processCamera(fragComp);
  processScene(fragComp);
}
} // namespace svulkan2
