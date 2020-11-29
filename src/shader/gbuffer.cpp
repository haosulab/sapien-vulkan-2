#include "svulkan2/shader/gbuffer.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {

void GbufferShaderConfig::parseGLSL(std::string const &vertFile,
                                    std::string const &fragFile) {
  GLSLCompiler compiler;
  mVertSpv = compiler.compileToSpirv(vk::ShaderStageFlagBits::eVertex,
                                     readFile(vertFile));
  log::info("shader compiled: " + vertFile);

  mFragSpv = compiler.compileToSpirv(vk::ShaderStageFlagBits::eFragment,
                                     readFile(fragFile));
  log::info("shader compiled: " + fragFile);

  reflectSPV();
}

void GbufferShaderConfig::reflectSPV() {
  spv_reflect::ShaderModule vertModule(mVertSpv);
  processVertexInput(vertModule);
}

void GbufferShaderConfig::processVertexInput(
    spv_reflect::ShaderModule &module) {
  uint32_t count;
  module.EnumerateInputVariables(&count, nullptr);
  std::vector<SpvReflectInterfaceVariable *> inputVariables(count);
  module.EnumerateInputVariables(&count, inputVariables.data());

  DataLayout vertexLayout;
  for (auto *var : inputVariables) {
    vertexLayout.elements[var->name] =
        DataLayoutElement{.name = var->name,
                          .typeName = GetTypeNameFromReflectFormat(var->format),
                          .size = GetSizeFromReflectFormat(var->format),
                          .location = var->location};
  }

  // check valid vertex format
  auto elements = vertexLayout.getElementsSorted();
  if (elements.size() == 0 || elements[0].name != "position" ||
      elements[0].typeName != "float3" || elements[0].location != 0) {
    throw std::runtime_error(
        "a float3 input variable named position bound at location "
        "0 is required for the gbuffer vertex shader");
  }
  std::vector<std::string> float3vars = {"normal", "tangent", "bitangent"};
  for (std::string key : float3vars) {
    if (vertexLayout.elements.find(key) != vertexLayout.elements.end()) {
      if (vertexLayout.elements[key].typeName != "float3") {
        throw std::runtime_error("input variable \"" + key +
                                 "\" should have type \"float3\"");
      }
    }
  }
  if (vertexLayout.elements.find("uv") != vertexLayout.elements.end()) {
    if (vertexLayout.elements["uv"].typeName != "float2") {
      throw std::runtime_error(
          "input variable \"uv\" should have type \"float2\"");
    }
  }
  if (vertexLayout.elements.find("color") != vertexLayout.elements.end()) {
    if (vertexLayout.elements["color"].typeName != "float4") {
      throw std::runtime_error(
          "input variable \"color\" should have type \"float4\"");
    }
  }
}

} // namespace svulkan2
