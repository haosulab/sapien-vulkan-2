#include "svulkan2/shader/gbuffer.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {

void GbufferPassParser::reflectSPV() {
    std::vector<uint32_t> spirv = mVertSPVCode;
    spirv_cross::Compiler vertComp(std::move(spirv));
    processVertexInput(vertComp);
    processCamera(vertComp);
    processObject(vertComp);
    spirv_cross::Compiler fragComp(mFragSPVCode);
    processMaterial(fragComp);
    processOutput(fragComp);
}

void GbufferPassParser::processVertexInput(spirv_cross::Compiler &compiler) {
  spirv_cross::ShaderResources resources = compiler.get_shader_resources();
  auto vertInput = resources.stage_inputs;

  bool usePosition = false;
  bool useNormal = false;
  bool useTangent = false;
  bool useBitangent = false;
  bool useColor = false;
  bool useUV = false;

  for (auto &var : vertInput) {
    if (var.name == "position") {
      usePosition = true;
      if (!type_is_float3(compiler.get_type(var.type_id)) ||
          compiler.get_decoration(var.id,
                                  spv::Decoration::DecorationLocation) != 0) {
        throw std::runtime_error(
            "gbuffer.vert: a float3 input variable named position bound at "
            "location 0 is required for the gbuffer vertex shader");
      }
      mVertexLayout.elements["position"] = {
          .name = "position",
          .type = DataType::eFLOAT3,
          .size = GetDataTypeSize(DataType::eFLOAT3),
          .location = 0};

    } else if (var.name == "normal") {
      useNormal = true;
      if (!type_is_float3(compiler.get_type(var.type_id))) {
        throw std::runtime_error("gbuffer.vert: normal input must be float3");
      }
      mVertexLayout.elements["normal"] = {
          .name = "normal",
          .type = DataType::eFLOAT3,
          .size = GetDataTypeSize(DataType::eFLOAT3),
          .location = compiler.get_decoration(
              var.id, spv::Decoration::DecorationLocation)};
    } else if (var.name == "tangent") {
      useTangent = true;
      if (!type_is_float3(compiler.get_type(var.type_id))) {
        throw std::runtime_error("gbuffer.vert: tangent input must be float3");
      }
      mVertexLayout.elements["tangent"] = {
          .name = "tangent",
          .type = DataType::eFLOAT3,
          .size = GetDataTypeSize(DataType::eFLOAT3),
          .location = compiler.get_decoration(
              var.id, spv::Decoration::DecorationLocation)};
    } else if (var.name == "bitangent") {
      useBitangent = true;
      if (!type_is_float3(compiler.get_type(var.type_id))) {
        throw std::runtime_error(
            "gbuffer.vert: bitangent input must be float3");
      }
      mVertexLayout.elements["bitangent"] = {
          .name = "bitangent",
          .type = DataType::eFLOAT3,
          .size = GetDataTypeSize(DataType::eFLOAT3),
          .location = compiler.get_decoration(
              var.id, spv::Decoration::DecorationLocation)};
    } else if (var.name == "uv") {
      useUV = true;
      if (!type_is_float2(compiler.get_type(var.type_id))) {
        throw std::runtime_error("gbuffer.vert: uv input must be float2");
      }
      mVertexLayout.elements["uv"] = {
          .name = "uv",
          .type = DataType::eFLOAT2,
          .size = GetDataTypeSize(DataType::eFLOAT2),
          .location = compiler.get_decoration(
              var.id, spv::Decoration::DecorationLocation)};
    } else if (var.name == "color") {
      useColor = true;
      if (!type_is_float4(compiler.get_type(var.type_id))) {
        throw std::runtime_error("gbuffer.vert: color input must be float4");
      }
      mVertexLayout.elements["color"] = {
          .name = "color",
          .type = DataType::eFLOAT4,
          .size = GetDataTypeSize(DataType::eFLOAT4),
          .location = compiler.get_decoration(
              var.id, spv::Decoration::DecorationLocation)};
    }
  }
  if (!usePosition) {
    throw std::runtime_error("gbuffer.vert: vertex position is required.");
  }
  if (useTangent && !useNormal) {
    throw std::runtime_error(
        "gbuffer.vert: vertex normal is required for vertex tangent.");
  }
  if (useBitangent && (!useNormal || !useTangent)) {
    throw std::runtime_error("gbuffer.vert: vertex normal and tangent are "
                             "required for vertex bitangent.");
  }
  log::info(mVertexLayout.summarize());
}

void GbufferPassParser::processCamera(spirv_cross::Compiler &compiler) {
  mCameraLayout = parseCamera(compiler, 0, 1, "gbuffer.vert: ");
}

void GbufferPassParser::processObject(spirv_cross::Compiler &compiler) {
  static const char *ERROR =
      "gbuffer.vert: Object must be specified by: layout(set=2, binding=0) "
      "uniform "
      "ObjectBuffer { mat4 modelMatrix; uvec4 segmentation; [optional]mat4 "
      "userData; } objectBuffer;";

  spirv_cross::ShaderResources resources = compiler.get_shader_resources();
  auto *binding = find_uniform_by_decoration(compiler, resources, 0, 2);

  if (!binding) {
    throw std::runtime_error(ERROR);
  }
  auto &type = compiler.get_type(binding->type_id);
  if (type.member_types.size() < 2 || type.member_types.size() > 3) {
    throw std::runtime_error(ERROR);
  }
  if (!type_is_float44(compiler.get_type(type.member_types[0])) ||
      !type_is_uint4(compiler.get_type(type.member_types[1]))) {
    throw std::runtime_error(ERROR);
  };
  if (compiler.get_member_name(type.self, 0) != "modelMatrix" ||
      compiler.get_member_name(type.self, 1) != "segmentation") {
    throw std::runtime_error(ERROR);
  };
  mObjectLayout.size = compiler.get_declared_struct_size(type);
  mObjectLayout.elements["modelMatrix"] = StructDataLayoutElement{
      .name = "modelMatrix",
      .type = DataType::eFLOAT44,
      .size = static_cast<uint32_t>(
          compiler.get_declared_struct_member_size(type, 0)),
      .offset = compiler.type_struct_member_offset(type, 0)};
  mObjectLayout.elements["segmentation"] = StructDataLayoutElement{
      .name = "segmentation",
      .type = DataType::eUINT4,
      .size = static_cast<uint32_t>(
          compiler.get_declared_struct_member_size(type, 1)),
      .offset = compiler.type_struct_member_offset(type, 1)};

  if (type.member_types.size() == 3) {
    if (!type_is_float44(compiler.get_type(type.member_types[2]))) {
      throw std::runtime_error(ERROR);
    }
    if (compiler.get_member_name(type.self, 2) != "userData") {
      throw std::runtime_error(ERROR);
    }
    mObjectLayout.elements["userData"] = StructDataLayoutElement{
        .name = "userData",
        .type = DataType::eFLOAT44,
        .size = static_cast<uint32_t>(
            compiler.get_declared_struct_member_size(type, 2)),
        .offset = compiler.type_struct_member_offset(type, 2)};
  }
  log::info(mObjectLayout.summarize());
}

void GbufferPassParser::processMaterial(spirv_cross::Compiler &compiler) {
  static const char *ERROR =
      "gbuffer.frag: Material must be specified at layout(set=2, binding=0) "
      "with specular or metallic pipeline.";

  auto resources = compiler.get_shader_resources();
  auto binding = find_uniform_by_decoration(compiler, resources, 0, 3);
  auto &type = compiler.get_type(binding->type_id);

  if (!binding) {
    throw std::runtime_error(ERROR);
  }
  if (type.member_types.size() == 6) {
    mMaterialType = eMETALLIC;
  } else if (type.member_types.size() == 6) {
    mMaterialType = eSPECULAR;
  } else {
    throw std::runtime_error(ERROR);
  }
  if (mMaterialType == eMETALLIC) {
    if (!type_is_float4(compiler.get_type(type.member_types[0])) ||
        !type_is_float(compiler.get_type(type.member_types[1])) ||
        !type_is_float(compiler.get_type(type.member_types[2])) ||
        !type_is_float(compiler.get_type(type.member_types[3])) ||
        !type_is_float(compiler.get_type(type.member_types[4])) ||
        !type_is_int(compiler.get_type(type.member_types[5]))) {
      throw std::runtime_error(ERROR);
    }
    if (compiler.get_member_name(type.self, 0) != "baseColor" ||
        compiler.get_member_name(type.self, 1) != "fresnel" ||
        compiler.get_member_name(type.self, 2) != "roughness" ||
        compiler.get_member_name(type.self, 3) != "metallic" ||
        compiler.get_member_name(type.self, 4) != "transparency" ||
        compiler.get_member_name(type.self, 5) != "textureMask") {
      throw std::runtime_error(ERROR);
    }

    // validate texture bindings
    std::vector textureBindings = {
        find_sampler_by_decoration(compiler, resources, 1, 3),
        find_sampler_by_decoration(compiler, resources, 2, 3),
        find_sampler_by_decoration(compiler, resources, 3, 3),
        find_sampler_by_decoration(compiler, resources, 4, 3)};
    for (auto texture : textureBindings) {
      if (!texture) {
        throw std::runtime_error(
            "gbuffer.frag: metallic material requires 4 textures, "
            "colorTexture, "
            "roughnessTexture, normalTexture, metallicTexture");
      }
    }
    if (textureBindings[0]->name != "colorTexture" ||
        textureBindings[1]->name != "roughnessTexture" ||
        textureBindings[2]->name != "normalTexture" ||
        textureBindings[3]->name != "metallicTexture") {
      throw std::runtime_error(
          "gbuffer.frag: metallic material requires 4 textures, "
          "colorTexture, "
          "roughnessTexture, normalTexture, metallicTexture");
    }

  } else if (mMaterialType == eSPECULAR) {
    if (!type_is_float4(compiler.get_type(type.member_types[0])) ||
        !type_is_float(compiler.get_type(type.member_types[1])) ||
        !type_is_float(compiler.get_type(type.member_types[2])) ||
        !type_is_int(compiler.get_type(type.member_types[3]))) {
      throw std::runtime_error(ERROR);
    }
    if (compiler.get_member_name(type.self, 0) != "diffuse" ||
        compiler.get_member_name(type.self, 1) != "specular" ||
        compiler.get_member_name(type.self, 2) != "transparency" ||
        compiler.get_member_name(type.self, 3) != "textureMask") {
      throw std::runtime_error(ERROR);
    }

    // validate texture bindings
    std::vector textureBindings = {
        find_sampler_by_decoration(compiler, resources, 1, 3),
        find_sampler_by_decoration(compiler, resources, 2, 3),
        find_sampler_by_decoration(compiler, resources, 3, 3),
    };
    for (auto texture : textureBindings) {
      if (!texture) {
        throw std::runtime_error(
            "gbuffer.frag: specular material requires 3 textures, "
            "colorTexture, specularTexture, normalTexture");
      }
    }
    if (textureBindings[0]->name != "colorTexture" ||
        textureBindings[1]->name != "specularTexture" ||
        textureBindings[2]->name != "normalTexture") {
      throw std::runtime_error(
          "gbuffer.frag: specular material requires 3 textures, "
          "colorTexture, specularTexture, normalTexture");
    }
  }
}

void GbufferPassParser::processOutput(spirv_cross::Compiler &compiler) {
  auto resource = compiler.get_shader_resources();
  auto outputs = resource.stage_outputs;

  for (auto &var : outputs) {
    if (var.name.size() <= 3 || var.name.substr(0, 3) != "out") {
      throw std::runtime_error(
          "gbuffer.frag: all output variable should start with \"out\" and "
          "be followed by a meaningful name");
    }
    std::string name = var.name.substr(3);
    if (mOutputLayout.elements.find(name) != mOutputLayout.elements.end()) {
      throw std::runtime_error("gbuffer.frag: duplicated output variable " +
                               name);
    }
    auto &type = compiler.get_type(var.type_id);
    auto dataType = get_data_type(type);

    mOutputLayout.elements[name] = {
        .name = name,
        .type = dataType,
        .size = GetDataTypeSize(dataType),
        .location = compiler.get_decoration(
            var.id, spv::Decoration::DecorationLocation),
    };
  }
  spdlog::info(mOutputLayout.summarize());
}

} // namespace svulkan2
