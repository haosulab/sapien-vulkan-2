#pragma once
#include "glsl_compiler.h"
#include "reflect.h"
#include "svulkan2/common/err.h"
#include "svulkan2/common/fs.h"
#include "svulkan2/common/layout.h"

namespace svulkan2 {
namespace shader {

inline std::string
getOutTextureName(std::string variableName) { // remove "out" prefix
  return variableName.substr(3, std::string::npos);
}

std::shared_ptr<InputDataLayout>
parseInputData(spirv_cross::Compiler &compiler);
std::shared_ptr<InputDataLayout>
parseVertexInput(spirv_cross::Compiler &compiler);

std::shared_ptr<OutputDataLayout>
parseOutputData(spirv_cross::Compiler &compiler);
std::shared_ptr<OutputDataLayout>
parseTextureOutput(spirv_cross::Compiler &compiler);

std::shared_ptr<StructDataLayout> parseBuffer(spirv_cross::Compiler &compiler,
                                              uint32_t bindingNumber,
                                              uint32_t setNumber);
std::shared_ptr<StructDataLayout>
parseCameraBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                  uint32_t setNumber);
std::shared_ptr<StructDataLayout>
parseMaterialBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                    uint32_t setNumber);
std::shared_ptr<StructDataLayout>
parseObjectBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                  uint32_t setNumber);
std::shared_ptr<StructDataLayout>
parseSceneBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                 uint32_t setNumber);

std::shared_ptr<CombinedSamplerLayout>
parseCombinedSampler(spirv_cross::Compiler &compiler);

std::shared_ptr<SpecializationConstantLayout>
parseSpecializationConstant(spirv_cross::Compiler &compiler);

class BaseParser {
protected:
  std::vector<uint32_t> mVertSPVCode;
  std::vector<uint32_t> mFragSPVCode;
  vk::UniquePipelineLayout mPipelineLayout;

public:
  void loadGLSLFiles(std::string const &vertFile, std::string const &fragFile);
  void loadSPVFiles(std::string const &vertFile, std::string const &fragFile);
  void loadSPVCode(std::vector<uint32_t> const &vertCode,
                   std::vector<uint32_t> const &fragCode);
  vk::PipelineLayout getPipelineLayout() const { return mPipelineLayout.get(); }

  virtual std::shared_ptr<OutputDataLayout> getTextureOutputLayout() const = 0;
  virtual std::vector<std::string> getRenderTargetNames() const = 0;
  virtual vk::RenderPass getRenderPass() const = 0;
  virtual vk::Pipeline getPipeline() const = 0;

protected:
  virtual void reflectSPV() = 0;
};
} // namespace shader
} // namespace svulkan2
