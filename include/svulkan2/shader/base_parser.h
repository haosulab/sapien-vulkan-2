#pragma once
#include "glsl_compiler.h"
#include "reflect.h"
#include "svulkan2/common/fs.h"
#include "svulkan2/common/layout.h"

namespace svulkan2 {
namespace shader {

inline void ASSERT(bool condition, std::string const &error) {
  if (!condition) {
    throw std::runtime_error(error);
  }
}

template <typename Container, typename T>
inline bool CONTAINS(Container &container, T const &element) {
  return container.find(element) != container.end();
}

std::unique_ptr<InputDataLayout>
parseInputData(spirv_cross::Compiler &compiler);
std::unique_ptr<InputDataLayout>
parseVertexInput(spirv_cross::Compiler &compiler);

std::unique_ptr<OutputDataLayout>
parseOutputData(spirv_cross::Compiler &compiler);
std::unique_ptr<OutputDataLayout>
parseTextureOutput(spirv_cross::Compiler &compiler);

std::unique_ptr<StructDataLayout> parseBuffer(spirv_cross::Compiler &compiler,
                                              uint32_t bindingNumber,
                                              uint32_t setNumber);
std::unique_ptr<StructDataLayout>
parseCameraBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                  uint32_t setNumber);
std::unique_ptr<StructDataLayout>
parseMaterialBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                    uint32_t setNumber);
std::unique_ptr<StructDataLayout>
parseObjectBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                  uint32_t setNumber);
std::unique_ptr<StructDataLayout>
parseSceneBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                 uint32_t setNumber);

std::unique_ptr<CombinedSamplerLayout>
parseCombinedSampler(spirv_cross::Compiler &compiler);

std::unique_ptr<SpecializationConstantLayout>
parseSpecializationConstant(spirv_cross::Compiler &compiler);

class BaseParser {
protected:
  std::vector<uint32_t> mVertSPVCode;
  std::vector<uint32_t> mFragSPVCode;

public:
  void loadGLSLFiles(std::string const &vertFile, std::string const &fragFile);
  void loadSPVFiles(std::string const &vertFile, std::string const &fragFile);
  void loadSPVCode(std::vector<uint32_t> const &vertCode,
                   std::vector<uint32_t> const &fragCode);

protected:
  virtual void reflectSPV() = 0;
};
} // namespace shader
} // namespace svulkan2
