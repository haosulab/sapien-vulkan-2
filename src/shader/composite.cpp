#include "svulkan2/shader/composite.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {
namespace shader {

void CompositePassParser::reflectSPV() {
  spirv_cross::Compiler vertComp(mVertSPVCode);
  spirv_cross::Compiler fragComp(mFragSPVCode);
  try {
    mCombinedSamplerLayout = parseCombinedSampler(fragComp);
    mTextureOutputLayout = parseTextureOutput(fragComp);
  } catch (std::runtime_error const &err) {
    throw std::runtime_error("[frag]" + std::string(err.what()));
  }
}

} // namespace shader
} // namespace svulkan2
