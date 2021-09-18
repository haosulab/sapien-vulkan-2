#pragma once
#include "glsl_compiler.h"
#include "reflect.h"
#include "svulkan2/common/err.h"
#include "svulkan2/common/fs.h"
#include "svulkan2/common/layout.h"
#include <future>
#include <map>

namespace svulkan2 {
namespace shader {

struct SpecializationConstantValue {
  DataType dtype;
  union {
    int intValue;
    float floatValue;
  };
};

enum UniformBindingType {
  eScene,
  eObject,
  eCamera,
  eMaterial,
  eTextures,
  eLight,
  eNone,
  eUnknown
};

struct DescriptorSetDescription {
  struct Binding {
    std::string name;
    vk::DescriptorType type;
    uint32_t arrayIndex;
  };
  UniformBindingType type{eUnknown};
  std::vector<std::shared_ptr<StructDataLayout>> buffers;
  std::vector<std::string> samplers;
  std::map<uint32_t, Binding> bindings;

  // TODO: need a bit testing
  DescriptorSetDescription merge(DescriptorSetDescription const &other) const;
};

DescriptorSetDescription
getDescriptorSetDescription(spirv_cross::Compiler &compiler,
                            uint32_t setNumber);

inline std::string
getOutTextureName(std::string variableName) { // remove "out" prefix
  if (variableName.substr(0, 3) != "out") {
    throw std::runtime_error("Output texture must start with \"out\"");
  }
  return variableName.substr(3, std::string::npos);
}

inline static std::string
getInTextureName(std::string variableName) { // remove "sampler" prefix
  if (variableName.substr(0, 7) != "sampler") {
    throw std::runtime_error("Input texture must start with \"sampler\"");
  }
  return variableName.substr(7, std::string::npos);
}

std::shared_ptr<InputDataLayout>
parseInputData(spirv_cross::Compiler &compiler);
std::shared_ptr<InputDataLayout>
parseVertexInput(spirv_cross::Compiler &compiler);

std::shared_ptr<OutputDataLayout>
parseOutputData(spirv_cross::Compiler &compiler);
std::shared_ptr<OutputDataLayout>
parseTextureOutput(spirv_cross::Compiler &compiler);

bool hasUniformBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                      uint32_t setNumber);

std::shared_ptr<StructDataLayout> parseBuffer(spirv_cross::Compiler &compiler,
                                              uint32_t bindingNumber,
                                              uint32_t setNumber);
std::shared_ptr<StructDataLayout> parseBuffer(spirv_cross::Compiler &compiler,
                                              spirv_cross::Resource &resource);
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
std::shared_ptr<StructDataLayout>
parseLightSpaceBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                      uint32_t setNumber);

std::shared_ptr<SpecializationConstantLayout>
parseSpecializationConstant(spirv_cross::Compiler &compiler);

class BaseParser {

protected:
  std::string mName;
  std::vector<uint32_t> mVertSPVCode;
  std::vector<uint32_t> mFragSPVCode;
  vk::UniquePipelineLayout mPipelineLayout;

  bool mAlphaBlend{};

public:
  void loadGLSLFiles(std::string const &vertFile, std::string const &fragFile);
  void loadSPVFiles(std::string const &vertFile, std::string const &fragFile);
  void loadSPVCode(std::vector<uint32_t> const &vertCode,
                   std::vector<uint32_t> const &fragCode);

  std::future<void> loadGLSLFilesAsync(std::string const &vertFile,
                                       std::string const &fragFile);

  vk::PipelineLayout getPipelineLayout() const { return mPipelineLayout.get(); }

  virtual std::shared_ptr<OutputDataLayout> getTextureOutputLayout() const = 0;
  virtual std::vector<std::string> getColorRenderTargetNames() const = 0;
  virtual std::optional<std::string> getDepthRenderTargetName() const {
    return {};
  };

  virtual vk::RenderPass getRenderPass() const = 0;
  virtual vk::Pipeline getPipeline() const = 0;
  virtual std::vector<UniformBindingType> getUniformBindingTypes() const;

  /** name of textures used in the texture descriptor set (only in
   * deferred-like passes) */
  virtual std::vector<std::string> getInputTextureNames() const;

  inline void enableAlphaBlend(bool enable) { mAlphaBlend = enable; }
  inline void setName(std::string const &name) { mName = name; }
  inline std::string getName() const { return mName; }

  virtual vk::Pipeline createGraphicsPipeline(
      vk::Device device, vk::Format colorFormat, vk::Format depthFormat,
      vk::CullModeFlags cullMode, vk::FrontFace frontFace,
      std::vector<std::pair<vk::ImageLayout, vk::ImageLayout>> const
          &colorTargetLayouts,
      std::pair<vk::ImageLayout, vk::ImageLayout> const &depthLayout,
      std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts,
      std::map<std::string, SpecializationConstantValue> const
          &specializationConstantInfo) = 0;

  virtual std::vector<DescriptorSetDescription>
  getDescriptorSetDescriptions() const = 0;

  virtual ~BaseParser() = default;

protected:
  virtual void reflectSPV() = 0;
};
} // namespace shader
} // namespace svulkan2
