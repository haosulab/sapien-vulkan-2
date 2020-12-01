#pragma once
#include "base_parser.h"

namespace svulkan2 {

class GbufferPassParser : public BaseParser {

  InOutDataLayout mVertexLayout{};
  StructDataLayout mCameraLayout{};
  StructDataLayout mObjectLayout{};
  InOutDataLayout mOutputLayout{};

  enum { eSPECULAR, eMETALLIC } mMaterialType;

private:
  void reflectSPV() override;

  void processVertexInput(spirv_cross::Compiler &module);
  void processCamera(spirv_cross::Compiler &vertModule);
  void processObject(spirv_cross::Compiler &vertModule);
  void processMaterial(spirv_cross::Compiler &fragModule);
  void processOutput(spirv_cross::Compiler &fragModule);
};

} // namespace svulkan2
