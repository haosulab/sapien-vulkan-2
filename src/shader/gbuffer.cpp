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
  processCamera(vertModule);
  processObject(vertModule);

  spv_reflect::ShaderModule fragModule(mFragSpv);
  processMaterial(fragModule);
  processOutput(fragModule);
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
  mVertexLayout = vertexLayout;
}

void GbufferShaderConfig::processCamera(spv_reflect::ShaderModule &vertModule) {
  static const char *ERROR =
      "gbuffer.vert: Camera must be specified by (the types are enforced but "
      "variable names "
      "are not): layout(set=1, binding=0) uniform CameraBuffer { mat4 "
      "viewMatrix; mat4 projectionMatrix; mat4 viewMatrixInverse; mat4 "
      "projectionMatrixInverse; } cameraBuffer;";

  static const char *WARN =
      "gbuffer.vert: GBuffer Camera is recommended to be specified by: "
      "layout(set=1, "
      "binding=0) uniform "
      "CameraBuffer { mat4 viewMatrix; mat4 projectionMatrix; mat4 "
      "viewMatrixInverse; mat4 projectionMatrixInverse; } cameraBuffer;";

  SpvReflectDescriptorBinding const *binding =
      vertModule.GetDescriptorBinding(0, 1);

  // validate camera
  if (!binding) {
    throw std::runtime_error(ERROR);
  }
  if (std::string(binding->name) != "cameraBuffer") {
    log::warn(WARN);
  }
  if (binding->block.member_count != 4 ||
      binding->block.members[0].size != 64 ||
      binding->block.members[1].size != 64 ||
      binding->block.members[2].size != 64 ||
      binding->block.members[3].size != 64) {
    throw std::runtime_error(ERROR);
  }

  if (std::string(binding->block.members[0].name) != "viewMatrix" ||
      std::string(binding->block.members[1].name) != "projectionMatrix" ||
      std::string(binding->block.members[2].name) != "viewMatrixInverse" ||
      std::string(binding->block.members[3].name) !=
          "projectionMatrixInverse") {
    throw std::runtime_error(ERROR);
  }

  mCameraLayout.size = binding->block.size;
  mCameraLayout.elements["viewMatrix"] =
      DataLayoutElement{.name = "viewMatrix",
                        .typeName = "float44",
                        .size = binding->block.members[0].size,
                        .location = binding->block.members[0].offset};
  mCameraLayout.elements["projectionMatrix"] =
      DataLayoutElement{.name = "projectionMatrix",
                        .typeName = "float44",
                        .size = binding->block.members[1].size,
                        .location = binding->block.members[1].offset};
  mCameraLayout.elements["viewMatrixInverse"] =
      DataLayoutElement{.name = "viewMatrixInverse",
                        .typeName = "float44",
                        .size = binding->block.members[2].size,
                        .location = binding->block.members[2].offset};
  mCameraLayout.elements["projectionMatrixInverse"] =
      DataLayoutElement{.name = "projectionMatrixInverse",
                        .typeName = "float44",
                        .size = binding->block.members[3].size,
                        .location = binding->block.members[3].offset};
  log::info(mCameraLayout.summarize());
}

void GbufferShaderConfig::processObject(spv_reflect::ShaderModule &vertModule) {
  static const char *ERROR =
      "gbuffer.vert: Object must be specified by: layout(set=2, binding=0) "
      "uniform "
      "ObjectBuffer { mat4 modelMatrix; uvec4 segmentation; [optional]mat4 "
      "userData; } objectBuffer;";
  SpvReflectDescriptorBinding const *binding =
      vertModule.GetDescriptorBinding(0, 2);
  if (!binding) {
    throw std::runtime_error(ERROR);
  }
  if (binding->block.member_count < 2 || binding->block.member_count > 3) {
    throw std::runtime_error(ERROR);
  }
  if (binding->block.members[0].size != 64 ||
      binding->block.members[1].size != 16) {
    throw std::runtime_error(ERROR);
  }
  if (std::string(binding->block.members[0].name) != "modelMatrix" ||
      std::string(binding->block.members[1].name) != "segmentation") {
    throw std::runtime_error(ERROR);
  }
  mObjectLayout.size = binding->block.size;
  mObjectLayout.elements["modelMatrix"] =
      DataLayoutElement{.name = "modelMatrix",
                        .typeName = "float44",
                        .size = binding->block.members[0].size,
                        .location = binding->block.members[0].offset};
  mObjectLayout.elements["segmentation"] =
      DataLayoutElement{.name = "segmentation",
                        .typeName = "uint4",
                        .size = binding->block.members[1].size,
                        .location = binding->block.members[1].offset};

  if (binding->block.member_count == 3) {
    if (binding->block.members[2].size != 64) {
      throw std::runtime_error(ERROR);
    }
    if (std::string(binding->block.members[2].name) != "userData") {
      throw std::runtime_error(ERROR);
    }
    mObjectLayout.elements["userData"] =
        DataLayoutElement{.name = "userData",
                          .typeName = "float44",
                          .size = binding->block.members[2].size,
                          .location = binding->block.members[2].offset};
  }
  log::info(mObjectLayout.summarize());
}

void GbufferShaderConfig::processMaterial(
    spv_reflect::ShaderModule &fragModule) {
  static const char *ERROR =
      "gbuffer.frag: Material must be specified at layout(set=2, binding=0) "
      "with specular or metallic pipeline.";

  auto binding = fragModule.GetDescriptorBinding(0, 3);
  if (!binding) {
    throw std::runtime_error(ERROR);
  }
  if (binding->block.member_count == 6) {
    mMaterialType = eMETALLIC;
  } else if (binding->block.member_count == 4) {
    mMaterialType = eSPECULAR;
  } else {
    throw std::runtime_error(ERROR);
  }
  if (mMaterialType == eMETALLIC) {
    if (binding->block.members[0].size != 16 ||
        binding->block.members[1].size != 4 ||
        binding->block.members[2].size != 4 ||
        binding->block.members[3].size != 4 ||
        binding->block.members[4].size != 4 ||
        binding->block.members[5].size != 4) {
      throw std::runtime_error(ERROR);
    }
    if (std::string(binding->block.members[0].name) != "baseColor" ||
        std::string(binding->block.members[1].name) != "fresnel" ||
        std::string(binding->block.members[2].name) != "roughness" ||
        std::string(binding->block.members[3].name) != "metallic" ||
        std::string(binding->block.members[4].name) != "transparency" ||
        std::string(binding->block.members[5].name) != "textureMask") {
      throw std::runtime_error(ERROR);
    }

    // validate texture bindings
    std::vector textureBindings = {
        fragModule.GetDescriptorBinding(1, 3),
        fragModule.GetDescriptorBinding(2, 3),
        fragModule.GetDescriptorBinding(3, 3),
        fragModule.GetDescriptorBinding(4, 3),
    };
    for (auto binding : textureBindings) {
      if (!binding || binding->descriptor_type !=
                          SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
        throw std::runtime_error(
            "gbuffer.frag: metallic material requires 4 textures, "
            "colorTexture, "
            "roughnessTexture, normalTexture, metallicTexture");
      }
    }
    if (std::string(textureBindings[0]->name) != "colorTexture" ||
        std::string(textureBindings[1]->name) != "roughnessTexture" ||
        std::string(textureBindings[2]->name) != "normalTexture" ||
        std::string(textureBindings[3]->name) != "metallicTexture") {
      throw std::runtime_error(
          "gbuffer.frag: metallic material requires 4 textures, colorTexture, "
          "roughnessTexture, normalTexture, metallicTexture");
    }

  } else if (mMaterialType == eMETALLIC) {
    if (binding->block.members[0].size != 16 ||
        binding->block.members[1].size != 16 ||
        binding->block.members[2].size != 4 ||
        binding->block.members[3].size != 4) {
      throw std::runtime_error(ERROR);
    }
    if (std::string(binding->block.members[0].name) != "diffuse" ||
        std::string(binding->block.members[1].name) != "specular" ||
        std::string(binding->block.members[2].name) != "transparency" ||
        std::string(binding->block.members[3].name) != "textureMask") {
      throw std::runtime_error(ERROR);
    }

    // validate texture bindings
    std::vector textureBindings = {
        fragModule.GetDescriptorBinding(1, 3),
        fragModule.GetDescriptorBinding(2, 3),
        fragModule.GetDescriptorBinding(3, 3),
    };
    for (auto binding : textureBindings) {
      if (!binding || binding->descriptor_type !=
                          SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
        throw std::runtime_error(
            "gbuffer.frag: specular material requires 3 textures, "
            "colorTexture, specularTexture, normalTexture");
      }
    }
    if (std::string(textureBindings[0]->name) != "colorTexture" ||
        std::string(textureBindings[1]->name) != "specularTexture" ||
        std::string(textureBindings[2]->name) != "normalTexture") {
      throw std::runtime_error(
          "gbuffer.frag: specular material requires 3 textures, "
          "colorTexture, specularTexture, normalTexture");
    }
  }
}

void GbufferShaderConfig::processOutput(spv_reflect::ShaderModule &fragModule) {
  uint32_t count;
  fragModule.EnumerateOutputVariables(&count, nullptr);

  std::vector<SpvReflectInterfaceVariable *> outputVariables(count);
  fragModule.EnumerateOutputVariables(&count, outputVariables.data());

  for (auto var : outputVariables) {
    std::string name = std::string(var->name);
    if (name.size() <= 3 || name.substr(0, 3) != "out") {
      throw std::runtime_error(
          "gbuffer.frag: all output variable should start with \"out\" and "
          "be followed by a meaningful name");
    }
    name = name.substr(3);
    if (mOutputLayout.elements.find(name) != mOutputLayout.elements.end()) {
      throw std::runtime_error("gbuffer.frag: duplicated output variable " + name);
    }
    mOutputLayout.elements[name] = {
      .name = name,
      .typeName = GetTypeNameFromReflectFormat(var->format),
      .size = GetSizeFromReflectFormat(var->format),
      .location = var->location,
    };
  }
  spdlog::info(mOutputLayout.summarize());
}

} // namespace svulkan2
