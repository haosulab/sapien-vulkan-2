#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace shader {

class GbufferPassParser : public BaseParser {
  std::shared_ptr<InputDataLayout> mVertexInputLayout;
  std::shared_ptr<StructDataLayout> mCameraBufferLayout;
  std::shared_ptr<StructDataLayout> mObjectBufferLayout;

  std::shared_ptr<StructDataLayout> mMaterialBufferLayout;
  std::shared_ptr<CombinedSamplerLayout> mCombinedSamplerLayout;
  std::shared_ptr<OutputDataLayout> mTextureOutputLayout;

public:
  enum { eSPECULAR, eMETALLIC } mMaterialType;

  std::vector<std::string> getOutputTextureNames() const;

private:
  void reflectSPV() override;
  void validate() const;
};

} // namespace shader
} // namespace svulkan2
