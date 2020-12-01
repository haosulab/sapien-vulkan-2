#pragma once
#include "base_parser.h"

namespace svulkan2 {
class DeferredPassParser : public BaseParser {
  InOutDataLayout mInputLayout{};
  InOutDataLayout mOutputLayout{};

  StructDataLayout mCameraLayout;

private:
  void processScene(spirv_cross::Compiler &compiler);
  void processCamera(spirv_cross::Compiler &compiler);
  void processInput(spirv_cross::Compiler &compiler);
  void processOutput(spirv_cross::Compiler &compiler);
  void reflectSPV() override;
};
} // namespace svulkan2
