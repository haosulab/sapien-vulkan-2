#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace shader {

class CompositePassParser : public BaseParser {
  std::unique_ptr<CombinedSamplerLayout> mCombinedSamplerLayout;
  std::unique_ptr<OutputDataLayout> mTextureOutputLayout;

public:
  inline CombinedSamplerLayout const &getCombinedSamplerLayout() const {
    return *mCombinedSamplerLayout;
  }
  inline OutputDataLayout const &getTextureOutputLayout() const {
    return *mTextureOutputLayout;
  }

private:
  void reflectSPV() override;
  void validate() const;
};

} // namespace shader
} // namespace svulkan2
