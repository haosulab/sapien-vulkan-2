#include "svulkan2/shader/deferred.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {
void DeferredPassParser::processInput(spirv_cross::Compiler &compiler) {}
void DeferredPassParser::processOutput(spirv_cross::Compiler &compiler) {}

void DeferredPassParser::processCamera(spirv_cross::Compiler &compiler) {
  mCameraLayout = parseCamera(compiler, 0, 1, "deferred.frag: ");
}

void DeferredPassParser::processScene(spirv_cross::Compiler &compiler) {
  const std::string ERROR1 =
      "deferred.frag: 2 specialization constants are required."
      "layout (constant_id = 0) const uint NUM_DIRECTIONAL_LIGHTS; "
      "layout (constant_id = 1) const uint NUM_POINT_LIGHTS;";

  const std::string ERROR2 =
      "deferred.frag: you need to define the following scene buffer."
      "struct PointLight {vec4 position; vec4 emission;}; "
      "struct DirectionalLight {vec4 direction; vec4 emission;}; layout(set = "
      "0, binding = 0) uniform SceneBuffer {vec4 ambientLight; "
      "DirectionalLight directionalLights[NUM_DIRECTIONAL_LIGHTS]; PointLight "
      "pointLights[NUM_POINT_LIGHTS];} sceneBuffer;";

  auto constants = compiler.get_specialization_constants();
  if (constants.size() != 2) {
    throw std::runtime_error(ERROR1);
  }
  for (auto &var : constants) {
    auto &constant = compiler.get_constant(var.id);
    if (var.constant_id == 0) {
      if (compiler.get_name(var.id) != "NUM_DIRECTIONAL_LIGHTS" ||
          !type_is_uint(compiler.get_type(constant.constant_type))) {
        throw std::runtime_error(ERROR1);
      }
      mNumDirectionalLights = constant.scalar_i32();
    } else if (var.constant_id == 1) {
      if (compiler.get_name(var.id) != "NUM_POINT_LIGHTS" ||
          !type_is_uint(compiler.get_type(constant.constant_type))) {
        throw std::runtime_error(ERROR1);
      }
      mNumPointLights = constant.scalar_i32();
    }
  }
  log::info("directional lights: {}; point lights: {}", mNumDirectionalLights,
            mNumPointLights);

  auto resources = compiler.get_shader_resources();
  auto binding = find_uniform_by_decoration(compiler, resources, 0, 0);
  if (!binding) {
    throw std::runtime_error(ERROR2);
  }
  auto &type = compiler.get_type(binding->type_id);
  if (type.member_types.size() != 3) {
    throw std::runtime_error(ERROR2);
  }
  if (!type_is_float4(compiler.get_type(type.member_types[0]))) {
    throw std::runtime_error(ERROR2);
  }
  auto &directionArrayType = compiler.get_type(type.member_types[1]);
  auto &pointArrayType = compiler.get_type(type.member_types[2]);

  if (directionArrayType.array_size_literal[0]) {
    throw std::runtime_error(ERROR2);
  };
  if (!type_is_float4(compiler.get_type(directionArrayType.member_types[0])) ||
      !type_is_float4(compiler.get_type(directionArrayType.member_types[1]))) {
    throw std::runtime_error(ERROR2);
  };
  if (directionArrayType.array_size_literal[1]) {
    throw std::runtime_error(ERROR2);
  };
  if (!type_is_float4(compiler.get_type(pointArrayType.member_types[0])) ||
      !type_is_float4(compiler.get_type(pointArrayType.member_types[1]))) {
    throw std::runtime_error(ERROR2);
  };

  mSceneLayout.size = compiler.get_declared_struct_size(type);
  mSceneLayout.elements["ambientLight"] = {
    .name = "ambientLight",
    .type = DataType::eFLOAT4,
    .size = static_cast<uint32_t>(
        compiler.get_declared_struct_member_size(type, 0)),
    .offset = compiler.type_struct_member_offset(type, 0)
  };
  mSceneLayout.elements["directionalLights"] = {
    .name = "directionalLights",
    .type = DataType::eUNKNOWN,
    .size = static_cast<uint32_t>(
        compiler.get_declared_struct_member_size(type, 1)),
    .offset = compiler.type_struct_member_offset(type, 1)
  };
  mSceneLayout.elements["pointLight"] = {
    .name = "pointLight",
    .type = DataType::eUNKNOWN,
    .size = static_cast<uint32_t>(
        compiler.get_declared_struct_member_size(type, 2)),
    .offset = compiler.type_struct_member_offset(type, 2)
  };
  log::info(mSceneLayout.summarize());
}

void DeferredPassParser::reflectSPV() {
  spirv_cross::Compiler vertComp(mVertSPVCode);
  spirv_cross::Compiler fragComp(mFragSPVCode);
  processCamera(fragComp);
  processScene(fragComp);
}
} // namespace svulkan2
