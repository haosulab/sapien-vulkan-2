#include "svulkan2/shader/base_parser.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {
void BaseParser::parseGLSLFiles(std::string const &vertFile,
                                std::string const &fragFile) {
  GLSLCompiler compiler;
  mVertSPVCode = compiler.compileToSpirv(vk::ShaderStageFlagBits::eVertex,
                                         readFile(vertFile));
  log::info("shader compiled: " + vertFile);

  mFragSPVCode = compiler.compileToSpirv(vk::ShaderStageFlagBits::eFragment,
                                         readFile(fragFile));
  log::info("shader compiled: " + fragFile);

  reflectSPV();
}

void BaseParser::parseSPVFiles(std::string const &vertFile,
                               std::string const &fragFile) {
  std::vector<char> vertCodeRaw = readFile(vertFile);
  std::vector<char> fragCodeRaw = readFile(fragFile);
  if (vertCodeRaw.size() / 4 * 4 != vertCodeRaw.size()) {
    throw std::runtime_error("corrupted SPV file: " + vertFile);
  }
  if (fragCodeRaw.size() / 4 * 4 != fragCodeRaw.size()) {
    throw std::runtime_error("corrupted SPV file: " + fragFile);
  }
  std::vector<uint32_t> vertCode(vertCodeRaw.size() / 4);
  std::vector<uint32_t> fragCode(fragCodeRaw.size() / 4);
  std::memcpy(vertCode.data(), vertCodeRaw.data(), vertCodeRaw.size());
  std::memcpy(fragCode.data(), fragCodeRaw.data(), fragCodeRaw.size());
  parseSPVCode(vertCode, fragCode);
}

void BaseParser::parseSPVCode(std::vector<uint32_t> const &vertCode,
                              std::vector<uint32_t> const &fragCode) {
  mVertSPVCode = vertCode;
  mFragSPVCode = fragCode;
  reflectSPV();
}

StructDataLayout BaseParser::parseCamera(spirv_cross::Compiler &compiler,
                                         uint32_t binding_number,
                                         uint32_t set_number,
                                         std::string errorPrefix) {
  const std::string ERROR =
      errorPrefix +
      "Camera must be specified by (the types are enforced but "
      "variable names "
      "are not): layout(set=1, binding=0) uniform CameraBuffer { mat4 "
      "viewMatrix; mat4 projectionMatrix; mat4 viewMatrixInverse; mat4 "
      "projectionMatrixInverse; } cameraBuffer;";

  const std::string WARN =
      errorPrefix +
      "GBuffer Camera is recommended to be specified by: "
      "layout(set=1, "
      "binding=0) uniform "
      "CameraBuffer { mat4 viewMatrix; mat4 projectionMatrix; mat4 "
      "viewMatrixInverse; mat4 projectionMatrixInverse; } cameraBuffer;";

  auto resources = compiler.get_shader_resources();

  auto binding = find_uniform_by_decoration(compiler, resources, binding_number,
                                            set_number);

  StructDataLayout layout;
  // validate camera
  if (!binding) {
    throw std::runtime_error(ERROR);
  }
  if (std::string(binding->name) != "cameraBuffer") {
    log::warn(WARN);
  }
  auto type = compiler.get_type(binding->type_id);

  if (type.member_types.size() != 4 ||
      !type_is_float44(compiler.get_type(type.member_types[0])) ||
      !type_is_float44(compiler.get_type(type.member_types[1])) ||
      !type_is_float44(compiler.get_type(type.member_types[2])) ||
      !type_is_float44(compiler.get_type(type.member_types[3]))) {
    throw std::runtime_error(ERROR);
  }

  if (compiler.get_member_name(type.self, 0) != "viewMatrix" ||
      compiler.get_member_name(type.self, 1) != "projectionMatrix" ||
      compiler.get_member_name(type.self, 2) != "viewMatrixInverse" ||
      compiler.get_member_name(type.self, 3) !=
          "projectionMatrixInverse") {
    throw std::runtime_error(ERROR);
  }

  layout.size = compiler.get_declared_struct_size(type);
  layout.elements["viewMatrix"] = StructDataLayoutElement{
      .name = "viewMatrix",
      .type = DataType::eFLOAT44,
      .size = static_cast<uint32_t>(
          compiler.get_declared_struct_member_size(type, 0)),
      .offset = compiler.type_struct_member_offset(type, 0)};
  layout.elements["projectionMatrix"] = StructDataLayoutElement{
      .name = "projectionMatrix",
      .type = DataType::eFLOAT44,
      .size = static_cast<uint32_t>(
          compiler.get_declared_struct_member_size(type, 1)),
      .offset = compiler.type_struct_member_offset(type, 1)};
  layout.elements["viewMatrixInverse"] = StructDataLayoutElement{
      .name = "viewMatrixInverse",
      .type = DataType::eFLOAT44,
      .size = static_cast<uint32_t>(
          compiler.get_declared_struct_member_size(type, 2)),
      .offset = compiler.type_struct_member_offset(type, 2)};
  layout.elements["projectionMatrixInverse"] = StructDataLayoutElement{
      .name = "projectionMatrixInverse",
      .type = DataType::eFLOAT44,
      .size = static_cast<uint32_t>(
          compiler.get_declared_struct_member_size(type, 3)),
      .offset = compiler.type_struct_member_offset(type, 2)};
  log::info(layout.summarize());
  return layout;
}

} // namespace svulkan2
