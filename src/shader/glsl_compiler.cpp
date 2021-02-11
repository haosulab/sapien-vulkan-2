#include "svulkan2/shader/glsl_compiler.h"
#include "svulkan2/common/log.h"
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <vulkan/vulkan.hpp>

namespace svulkan2 {

static TBuiltInResource GetDefaultTBuiltInResource() {
  TBuiltInResource Resources;

  Resources.maxLights = 32;
  Resources.maxClipPlanes = 6;
  Resources.maxTextureUnits = 32;
  Resources.maxTextureCoords = 32;
  Resources.maxVertexAttribs = 64;
  Resources.maxVertexUniformComponents = 4096;
  Resources.maxVaryingFloats = 64;
  Resources.maxVertexTextureImageUnits = 32;
  Resources.maxCombinedTextureImageUnits = 80;
  Resources.maxTextureImageUnits = 32;
  Resources.maxFragmentUniformComponents = 4096;
  Resources.maxDrawBuffers = 32;
  Resources.maxVertexUniformVectors = 128;
  Resources.maxVaryingVectors = 8;
  Resources.maxFragmentUniformVectors = 16;
  Resources.maxVertexOutputVectors = 16;
  Resources.maxFragmentInputVectors = 15;
  Resources.minProgramTexelOffset = -8;
  Resources.maxProgramTexelOffset = 7;
  Resources.maxClipDistances = 8;
  Resources.maxComputeWorkGroupCountX = 65535;
  Resources.maxComputeWorkGroupCountY = 65535;
  Resources.maxComputeWorkGroupCountZ = 65535;
  Resources.maxComputeWorkGroupSizeX = 1024;
  Resources.maxComputeWorkGroupSizeY = 1024;
  Resources.maxComputeWorkGroupSizeZ = 64;
  Resources.maxComputeUniformComponents = 1024;
  Resources.maxComputeTextureImageUnits = 16;
  Resources.maxComputeImageUniforms = 8;
  Resources.maxComputeAtomicCounters = 8;
  Resources.maxComputeAtomicCounterBuffers = 1;
  Resources.maxVaryingComponents = 60;
  Resources.maxVertexOutputComponents = 64;
  Resources.maxGeometryInputComponents = 64;
  Resources.maxGeometryOutputComponents = 128;
  Resources.maxFragmentInputComponents = 128;
  Resources.maxImageUnits = 8;
  Resources.maxCombinedImageUnitsAndFragmentOutputs = 8;
  Resources.maxCombinedShaderOutputResources = 8;
  Resources.maxImageSamples = 0;
  Resources.maxVertexImageUniforms = 0;
  Resources.maxTessControlImageUniforms = 0;
  Resources.maxTessEvaluationImageUniforms = 0;
  Resources.maxGeometryImageUniforms = 0;
  Resources.maxFragmentImageUniforms = 8;
  Resources.maxCombinedImageUniforms = 8;
  Resources.maxGeometryTextureImageUnits = 16;
  Resources.maxGeometryOutputVertices = 256;
  Resources.maxGeometryTotalOutputComponents = 1024;
  Resources.maxGeometryUniformComponents = 1024;
  Resources.maxGeometryVaryingComponents = 64;
  Resources.maxTessControlInputComponents = 128;
  Resources.maxTessControlOutputComponents = 128;
  Resources.maxTessControlTextureImageUnits = 16;
  Resources.maxTessControlUniformComponents = 1024;
  Resources.maxTessControlTotalOutputComponents = 4096;
  Resources.maxTessEvaluationInputComponents = 128;
  Resources.maxTessEvaluationOutputComponents = 128;
  Resources.maxTessEvaluationTextureImageUnits = 16;
  Resources.maxTessEvaluationUniformComponents = 1024;
  Resources.maxTessPatchComponents = 120;
  Resources.maxPatchVertices = 32;
  Resources.maxTessGenLevel = 64;
  Resources.maxViewports = 16;
  Resources.maxVertexAtomicCounters = 0;
  Resources.maxTessControlAtomicCounters = 0;
  Resources.maxTessEvaluationAtomicCounters = 0;
  Resources.maxGeometryAtomicCounters = 0;
  Resources.maxFragmentAtomicCounters = 8;
  Resources.maxCombinedAtomicCounters = 8;
  Resources.maxAtomicCounterBindings = 1;
  Resources.maxVertexAtomicCounterBuffers = 0;
  Resources.maxTessControlAtomicCounterBuffers = 0;
  Resources.maxTessEvaluationAtomicCounterBuffers = 0;
  Resources.maxGeometryAtomicCounterBuffers = 0;
  Resources.maxFragmentAtomicCounterBuffers = 1;
  Resources.maxCombinedAtomicCounterBuffers = 1;
  Resources.maxAtomicCounterBufferSize = 16384;
  Resources.maxTransformFeedbackBuffers = 4;
  Resources.maxTransformFeedbackInterleavedComponents = 64;
  Resources.maxCullDistances = 8;
  Resources.maxCombinedClipAndCullDistances = 8;
  Resources.maxSamples = 4;
  Resources.maxMeshOutputVerticesNV = 256;
  Resources.maxMeshOutputPrimitivesNV = 512;
  Resources.maxMeshWorkGroupSizeX_NV = 32;
  Resources.maxMeshWorkGroupSizeY_NV = 1;
  Resources.maxMeshWorkGroupSizeZ_NV = 1;
  Resources.maxTaskWorkGroupSizeX_NV = 32;
  Resources.maxTaskWorkGroupSizeY_NV = 1;
  Resources.maxTaskWorkGroupSizeZ_NV = 1;
  Resources.maxMeshViewCountNV = 4;

  Resources.limits.nonInductiveForLoops = 1;
  Resources.limits.whileLoops = 1;
  Resources.limits.doWhileLoops = 1;
  Resources.limits.generalUniformIndexing = 1;
  Resources.limits.generalAttributeMatrixVectorIndexing = 1;
  Resources.limits.generalVaryingIndexing = 1;
  Resources.limits.generalSamplerIndexing = 1;
  Resources.limits.generalVariableIndexing = 1;
  Resources.limits.generalConstantMatrixVectorIndexing = 1;

  return Resources;
}

static EShLanguage GetEShLanguage(vk::ShaderStageFlagBits stage) {
  switch (stage) {
  case vk::ShaderStageFlagBits::eVertex:
    return EShLanguage::EShLangVertex;
  case vk::ShaderStageFlagBits::eFragment:
    return EShLanguage::EShLangFragment;
  default:
    throw std::invalid_argument("GetEShLanguage: invalid shader stage.");
  }
}

void GLSLCompiler::InitializeProcess() { glslang::InitializeProcess(); }
void GLSLCompiler::FinalizeProcess() { glslang::FinalizeProcess(); }

std::vector<std::uint32_t>
GLSLCompiler::compileToSpirv(vk::ShaderStageFlagBits shaderStage,
                             std::vector<char> const &glslCode) {

  EShLanguage language = GetEShLanguage(shaderStage);
  glslang::TShader shader(language);
  std::string code(glslCode.begin(), glslCode.end());
  const char *codes[1] = {code.c_str()};

  shader.setStrings(codes, 1);
  shader.setEnvInput(glslang::EShSourceGlsl, GetEShLanguage(shaderStage),
                     glslang::EShClientVulkan, 110);
  shader.setEnvClient(glslang::EShClientVulkan,
                      glslang::EShTargetClientVersion::EShTargetVulkan_1_1);
  shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_1);

  TBuiltInResource resource = GetDefaultTBuiltInResource();
  EShMessages messages = static_cast<EShMessages>(
      EShMsgDefault | EShMsgVulkanRules | EShMsgSpvRules);
  bool result = shader.parse(&resource, 110, false, messages);

  {
    std::string log = shader.getInfoLog();
    if (log.length()) {
      log::getLogger()->warn(log);
    }
    log = shader.getInfoDebugLog();
    if (log.length()) {
      log::getLogger()->warn(log);
    }
  }

  if (!result) {
    throw std::runtime_error("copmileToSpirv: failed to compile shader");
  }

  glslang::TProgram program;
  program.addShader(&shader);
  result = program.link(messages);
  {
    std::string log = program.getInfoLog();
    if (log.length()) {
      log::getLogger()->warn(log);
    }
    log = program.getInfoDebugLog();
    if (log.length()) {
      log::getLogger()->warn(log);
    }
  }
  if (!result) {
    throw std::runtime_error("copmileToSpirv: failed to link shader");
  }

  glslang::TIntermediate *intermediate = program.getIntermediate(language);
  if (!intermediate) {
    throw std::runtime_error("copmileToSpirv: failed to get intermediate");
  }
  spv::SpvBuildLogger logger;

  std::vector<std::uint32_t> spirv;
  glslang::GlslangToSpv(*intermediate, spirv, &logger);

  {
    std::string log = logger.getAllMessages();
    if (log.length()) {
      log::getLogger()->warn(log);
    }
  }

  return spirv;
}

} // namespace svulkan2
