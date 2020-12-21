#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace shader {
class DeferredPassParser : public BaseParser {

  std::shared_ptr<SpecializationConstantLayout> mSpecializationConstantLayout;
  std::shared_ptr<StructDataLayout> mCameraBufferLayout;
  std::shared_ptr<StructDataLayout> mSceneBufferLayout;
  std::shared_ptr<CombinedSamplerLayout> mCombinedSamplerLayout;
  std::shared_ptr<OutputDataLayout> mTextureOutputLayout;

public:
  inline std::shared_ptr<SpecializationConstantLayout>
  getSpecializationConstantLayout() const {
    return mSpecializationConstantLayout;
  }
  inline std::shared_ptr<StructDataLayout> getCameraBufferLayout() const {
    return mCameraBufferLayout;
  }
  inline std::shared_ptr<StructDataLayout> getSceneBufferLayout() const {
    return mSceneBufferLayout;
  }
  inline std::shared_ptr<CombinedSamplerLayout>
  getCombinedSamplerLayout() const {
    return mCombinedSamplerLayout;
  }
  inline std::shared_ptr<OutputDataLayout> getTextureOutputLayout() const {
    return mTextureOutputLayout;
  }

private:
  void reflectSPV() override;
  void validate() const;
};
} // namespace shader
} // namespace svulkan2
