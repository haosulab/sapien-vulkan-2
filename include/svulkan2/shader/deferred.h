#pragma once
#include "base_parser.h"

namespace svulkan2 {
class DeferredPassParser : public BaseParser {
  InOutDataLayout mInputLayout{};
  InOutDataLayout mOutputLayout{};
  StructDataLayout mCameraLayout{};
  StructDataLayout mSceneLayout{};

  uint32_t mNumDirectionalLights = 0;
  uint32_t mNumPointLights = 0;

private:
  void processScene(spirv_cross::Compiler &compiler);
  void processCamera(spirv_cross::Compiler &compiler);
  void processInput(spirv_cross::Compiler &compiler);
  void processOutput(spirv_cross::Compiler &compiler);
  void reflectSPV() override;
};
} // namespace svulkan2
