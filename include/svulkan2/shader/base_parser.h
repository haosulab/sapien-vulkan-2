#pragma once
#include "glsl_compiler.h"
#include "reflect.h"
#include "svulkan2/common/fs.h"
#include "svulkan2/common/layout.h"

namespace svulkan2 {

class BaseParser {
protected:
  std::vector<uint32_t> mVertSPVCode;
  std::vector<uint32_t> mFragSPVCode;

public:
  void parseGLSLFiles(std::string const &vertFile, std::string const &fragFile);
  void parseSPVFiles(std::string const &vertFile, std::string const &fragFile);
  void parseSPVCode(std::vector<uint32_t> const &vertCode,
                    std::vector<uint32_t> const &fragCode);

protected:
  virtual void reflectSPV() = 0;
  StructDataLayout parseCamera(spirv_cross::Compiler &compiler,
                               uint32_t binding, uint32_t set,
                               std::string errorPrefix);
};

} // namespace svulkan2
