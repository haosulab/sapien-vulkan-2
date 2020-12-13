#include "svulkan2/shader/gbuffer.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {
namespace shader {
void GbufferPassParser::reflectSPV() {
  spirv_cross::Compiler vertComp(mVertSPVCode);
  try {
    mVertexInputLayout = parseVertexInput(vertComp);
    mCameraBufferLayout = parseCameraBuffer(vertComp, 0, 1);
    mObjectBufferLayout = parseObjectBuffer(vertComp, 0, 2);
  } catch (std::runtime_error const &err) {
    throw std::runtime_error("[vert]" + std::string(err.what()));
  }

  spirv_cross::Compiler fragComp(mFragSPVCode);
  try {
    mMaterialBufferLayout = parseMaterialBuffer(fragComp, 0, 3);
    mCombinedSamplerLayout = parseCombinedSampler(fragComp);
    mTextureOutputLayout = parseTextureOutput(fragComp);
  } catch (std::runtime_error const &err) {
    throw std::runtime_error("[frag]" + std::string(err.what()));
  }

  validate();
}

void GbufferPassParser::validate() const {
  for (auto &elem : mCombinedSamplerLayout->elements) {
    ASSERT(elem.second.binding >= 1 && elem.second.set == 3,
           "[frag]all textures should be bound to set 3, binding >= 1");
  }
};

std::vector<std::string> GbufferPassParser::getOutputTextureNames() const {
  std::vector<std::string> result;
  auto elems = mTextureOutputLayout->getElementsSorted();
  for (auto &elem : elems) {
    result.push_back(elem.name.substr(3));
  }
  return result;
}

} // namespace shader
} // namespace svulkan2
