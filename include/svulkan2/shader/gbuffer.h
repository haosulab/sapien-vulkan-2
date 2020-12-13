#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace shader {

class GbufferPassParser : public BaseParser {
  std::unique_ptr<InputDataLayout> mVertexInputLayout;
  std::unique_ptr<StructDataLayout> mCameraBufferLayout;
  std::unique_ptr<StructDataLayout> mObjectBufferLayout;

  std::unique_ptr<StructDataLayout> mMaterialBufferLayout;
  std::unique_ptr<CombinedSamplerLayout> mCombinedSamplerLayout;
  std::unique_ptr<OutputDataLayout> mTextureOutputLayout;

public:
  enum { eSPECULAR, eMETALLIC } mMaterialType;

  std::vector<std::string> getOutputTextureNames() const;

private:
  void reflectSPV() override;
  void validate() const;
};

} // namespace shader
} // namespace svulkan2
