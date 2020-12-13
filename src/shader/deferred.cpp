#include "svulkan2/shader/deferred.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {
namespace shader {

void DeferredPassParser::reflectSPV() {
  spirv_cross::Compiler vertComp(mVertSPVCode);
  spirv_cross::Compiler fragComp(mFragSPVCode);
  try {
    mSpecializationConstantLayout = parseSpecializationConstant(fragComp);
    mCameraBufferLayout = parseCameraBuffer(fragComp, 0, 1);
    mSceneBufferLayout = parseSceneBuffer(fragComp, 0, 0);
    mCombinedSamplerLayout = parseCombinedSampler(fragComp);
    mTextureOutputLayout = parseTextureOutput(fragComp);
  } catch (std::runtime_error const &err) {
    throw std::runtime_error("[frag]" + std::string(err.what()));
  }
}

void DeferredPassParser::validate() const {
  // validate constants
  ASSERT(CONTAINS(mSpecializationConstantLayout->elements,
                  "NUM_DIRECTIONAL_LIGHTS"),
         "[frag]NUM_DIRECTIONAL_LIGHTS is a required specialization "
         "constant");
  ASSERT(CONTAINS(mSpecializationConstantLayout->elements, "NUM_POINT_LIGHTS"),
         "[frag]NUM_POINT_LIGHTS is a required specialization "
         "constant");

  // validate samplers
  for (auto &sampler : mCombinedSamplerLayout->elements) {
    ASSERT(
        sampler.second.name.length() > 7 &&
            sampler.second.name.substr(0, 7) == "sampler",
        "[frag]texture sampler variable must start with \"sampler\"");
    ASSERT(sampler.second.set == 2, "all deferred.frag: all texture sampler "
                                    "should be bound at descriptor set 2");
  }
}

} // namespace shader
} // namespace svulkan2
