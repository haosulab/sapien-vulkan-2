#include "svulkan2/shader/base_parser.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {
namespace shader {

DescriptorSetDescription
DescriptorSetDescription::merge(DescriptorSetDescription const &other) const {
  if (type != other.type) {
    throw std::runtime_error("Descriptor merge failed: different types");
  }
  DescriptorSetDescription newLayout;
  newLayout.type = type;

  std::vector<std::shared_ptr<StructDataLayout>> newBuffers = buffers;
  std::vector<std::string> newSamplers = samplers;
  std::map<uint32_t, Binding> newBindings = bindings;

  for (auto &[bindingIndex, binding] : other.bindings) {
    if (bindings.find(bindingIndex) != bindings.end()) {
      if (binding.type == vk::DescriptorType::eUniformBuffer) {
        if (*other.buffers[binding.arrayIndex] !=
            *buffers[bindings.at(bindingIndex).arrayIndex]) {
          throw std::runtime_error("Incompatible descriptor at binding " +
                                   std::to_string(bindingIndex) +
                                   ": buffers are different.");
        }
      } else if (binding.type == vk::DescriptorType::eCombinedImageSampler) {
        if (other.samplers[binding.arrayIndex] !=
            samplers[(bindings.at(bindingIndex)).arrayIndex]) {
          throw std::runtime_error("Incompatible descriptor at binding " +
                                   std::to_string(bindingIndex) +
                                   ": texture names are different.");
        }
      } else {
        throw std::runtime_error(
            "Descriptor merge failed: unsupported descriptor type");
      }
    } else {
      if (binding.type == vk::DescriptorType::eUniformBuffer) {
        newBuffers.push_back(other.buffers[binding.arrayIndex]);
        newBindings[bindingIndex] = {
            .name = binding.name,
            .type = vk::DescriptorType::eUniformBuffer,
            .dim = binding.dim,
            .arraySize = binding.arraySize,
            .arrayIndex = static_cast<uint32_t>(newBuffers.size() - 1)};
      } else if (binding.type == vk::DescriptorType::eCombinedImageSampler) {
        newSamplers.push_back(other.samplers[binding.arrayIndex]);
        newBindings[bindingIndex] = {
            .name = binding.name,
            .type = vk::DescriptorType::eCombinedImageSampler,
            .dim = binding.dim,
            .arraySize = binding.arraySize,
            .arrayIndex = static_cast<uint32_t>(newSamplers.size() - 1)};
      } else {
        throw std::runtime_error(
            "Descriptor merge failed: unsupported descriptor type");
      }
    }
  }
  newLayout.buffers = newBuffers;
  newLayout.samplers = newSamplers;
  newLayout.bindings = newBindings;
  return newLayout;
}

std::shared_ptr<InputDataLayout>
parseInputData(spirv_cross::Compiler &compiler) {
  auto layout = std::make_shared<InputDataLayout>();
  auto resource = compiler.get_shader_resources();
  auto inputs = resource.stage_inputs;

  for (auto &var : inputs) {
    if (layout->elements.find(var.name) != layout->elements.end()) {
      throw std::runtime_error("duplicate input variable " + var.name);
    }
    auto &type = compiler.get_type(var.type_id);
    auto dataType = get_data_type(type);

    layout->elements[var.name] = {
        .name = var.name,
        .location = compiler.get_decoration(
            var.id, spv::Decoration::DecorationLocation),
        .dtype = dataType};
  }
  return layout;
}

std::shared_ptr<OutputDataLayout>
parseOutputData(spirv_cross::Compiler &compiler) {
  auto layout = std::make_shared<OutputDataLayout>();
  auto resource = compiler.get_shader_resources();
  auto outputs = resource.stage_outputs;

  for (auto &var : outputs) {
    if (layout->elements.find(var.name) != layout->elements.end()) {
      throw std::runtime_error("duplicate output variable " + var.name);
    }
    auto &type = compiler.get_type(var.type_id);
    auto dataType = get_data_type(type);

    layout->elements[var.name] = {
        .name = var.name,
        .location = compiler.get_decoration(
            var.id, spv::Decoration::DecorationLocation),
        .dtype = dataType};
  }
  return layout;
}

std::shared_ptr<InputDataLayout>
parseVertexInput(spirv_cross::Compiler &compiler) {
  auto vertexData = parseInputData(compiler);
  // required attribute
  ASSERT(CONTAINS(vertexData->elements, "position"),
         "vertex position is required at location 0 with type float3");

  // required type
  ASSERT(vertexData->elements["position"].dtype == eFLOAT3,
         "The following is required: layout(location = 0) in vec3 position;");
  ASSERT(!CONTAINS(vertexData->elements, "normal") ||
             vertexData->elements["normal"].dtype == eFLOAT3,
         "normal in vertex input must be a float3");
  ASSERT(!CONTAINS(vertexData->elements, "uv") ||
             vertexData->elements["uv"].dtype == eFLOAT2,
         "uv in vertex input must be a float2");
  ASSERT(!CONTAINS(vertexData->elements, "tangent") ||
             vertexData->elements["tangent"].dtype == eFLOAT3,
         "tangent in vertex input must be a float3");
  ASSERT(!CONTAINS(vertexData->elements, "bitangent") ||
             vertexData->elements["bitangent"].dtype == eFLOAT3,
         "bitangent in vertex input must be a float3");
  ASSERT(!CONTAINS(vertexData->elements, "color") ||
             vertexData->elements["color"].dtype == eFLOAT4,
         "color in vertex input must be a float4");

  // required associated types
  ASSERT(!CONTAINS(vertexData->elements, "tangent") ||
             CONTAINS(vertexData->elements, "normal"),
         "normal in vertex input is required when using tangent");
  ASSERT(!CONTAINS(vertexData->elements, "bitangent") ||
             (CONTAINS(vertexData->elements, "normal") &&
              CONTAINS(vertexData->elements, "tangent")),
         "normal and tangent in vertex input are required when using tangent");

  return vertexData;
}

std::shared_ptr<OutputDataLayout>
parseTextureOutput(spirv_cross::Compiler &compiler) {
  auto textureOutput = parseOutputData(compiler);
  for (auto &elem : textureOutput->elements) {
    ASSERT(elem.second.name.substr(0, 3) == "out",
           "all output variables must start with \"out\"");
  }

  ASSERT(!CONTAINS(textureOutput->elements, "outAlbedo") ||
             textureOutput->elements["outAlbedo"].dtype == eFLOAT4,
         "outAlbedo must be float4");
  ASSERT(!CONTAINS(textureOutput->elements, "outPosition") ||
             textureOutput->elements["outPosition"].dtype == eFLOAT4,
         "outPosition must be float4");
  ASSERT(!CONTAINS(textureOutput->elements, "outNormal") ||
             textureOutput->elements["outNormal"].dtype == eFLOAT4,
         "outNormal must be float4");
  ASSERT(!CONTAINS(textureOutput->elements, "outSegmentation") ||
             textureOutput->elements["outSegmentation"].dtype == eUINT4,
         "outSegmentation must be uint4");

  return textureOutput;
}

std::shared_ptr<StructDataLayout>
parseBuffer(spirv_cross::Compiler &compiler,
            spirv_cross::SPIRType const &type) {
  auto layout = std::make_shared<StructDataLayout>();
  layout->size = compiler.get_declared_struct_size(type);

  for (uint32_t i = 0; i < type.member_types.size(); ++i) {
    std::string memberName = compiler.get_member_name(type.self, i);
    uint32_t memberOffset = compiler.type_struct_member_offset(type, i);
    auto &memberType = compiler.get_type(type.member_types[i]);
    uint32_t memberSize = compiler.get_declared_struct_member_size(type, i);
    DataType dataType = get_data_type(memberType);

    if (memberOffset + memberSize > layout->size) {
      layout->size = memberOffset + memberSize;
    }

    if (dataType == eSTRUCT) {
      layout->elements[memberName] = {
          .name = memberName,
          .size = memberSize,
          .offset = memberOffset,
          .arrayDim = static_cast<uint32_t>(memberType.array.size()),
          .dtype = dataType,
          .member = parseBuffer(compiler, memberType)};
    } else {
      layout->elements[memberName] = {
          .name = memberName,
          .size = memberSize,
          .offset = memberOffset,
          .arrayDim = static_cast<uint32_t>(memberType.array.size()),
          .dtype = dataType,
          .member = nullptr};
    }
  }
  return layout;
}

std::shared_ptr<StructDataLayout> parseBuffer(spirv_cross::Compiler &compiler,
                                              spirv_cross::Resource &resource) {
  auto &type = compiler.get_type(resource.type_id);
  return parseBuffer(compiler, type);
}

bool hasUniformBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                      uint32_t setNumber) {
  auto resources = compiler.get_shader_resources();
  auto binding =
      find_uniform_by_decoration(compiler, resources, bindingNumber, setNumber);
  return binding != nullptr;
}

std::shared_ptr<StructDataLayout> parseBuffer(spirv_cross::Compiler &compiler,
                                              uint32_t bindingNumber,
                                              uint32_t setNumber) {
  auto resources = compiler.get_shader_resources();
  auto binding =
      find_uniform_by_decoration(compiler, resources, bindingNumber, setNumber);

  if (!binding) {
    throw std::runtime_error(
        "no buffer bound at binding=" + std::to_string(bindingNumber) +
        " set=" + std::to_string(setNumber));
  }
  auto &type = compiler.get_type(binding->type_id);
  return parseBuffer(compiler, type);
}

void verifyCameraBuffer(std::shared_ptr<StructDataLayout> layout) {
  // required fields
  ASSERT(CONTAINS(layout->elements, "viewMatrix"),
         "camera buffer requires viewMatrix");
  ASSERT(CONTAINS(layout->elements, "viewMatrixInverse"),
         "camera buffer requires viewMatrixInverse");
  ASSERT(CONTAINS(layout->elements, "projectionMatrix"),
         "camera buffer requires projectionMatrix");
  ASSERT(CONTAINS(layout->elements, "projectionMatrixInverse"),
         "camera buffer requires projectionMatrixInverse");

  // required types
  ASSERT(layout->elements["viewMatrix"].dtype == eFLOAT44,
         "camera viewMatrix should have type float44");
  ASSERT(layout->elements["viewMatrixInverse"].dtype == eFLOAT44,
         "camera viewMatrixInverse should have type float44");
  ASSERT(layout->elements["projectionMatrix"].dtype == eFLOAT44,
         "camera projectionMatrix should have type float44");
  ASSERT(layout->elements["projectionMatrixInverse"].dtype == eFLOAT44,
         "camera projectionMatrixInverse should have type float44");
  if (CONTAINS(layout->elements, "prevViewMatrix")) {
    ASSERT(layout->elements["prevViewMatrix"].dtype == eFLOAT44,
           "camera prevViewMatrix should have type float44");
  }
  if (CONTAINS(layout->elements, "prevViewMatrixInverse")) {
    ASSERT(layout->elements["prevViewMatrixInverse"].dtype == eFLOAT44,
           "camera prevViewMatrixInverse should have type float44");
  }
  if (CONTAINS(layout->elements, "width")) {
    ASSERT(layout->elements["width"].dtype == eFLOAT,
           "camera width should have type float");
  }
  if (CONTAINS(layout->elements, "height")) {
    ASSERT(layout->elements["height"].dtype == eFLOAT,
           "camera height should have type float");
  }
}

void verifyMaterialBuffer(std::shared_ptr<StructDataLayout> layout) {
  ASSERT(layout->elements.size() == 7,
         "Material should contain exactly 7 member variables");
  // required variables
  ASSERT(CONTAINS(layout->elements, "emission"),
         "material requires variable emission");
  ASSERT(CONTAINS(layout->elements, "baseColor"),
         "material requires variable baseColor");
  ASSERT(CONTAINS(layout->elements, "fresnel"),
         "material requires variable fresnel");
  ASSERT(CONTAINS(layout->elements, "roughness"),
         "material requires variable roughness");
  ASSERT(CONTAINS(layout->elements, "metallic"),
         "material requires variable metallic");
  ASSERT(CONTAINS(layout->elements, "transparency"),
         "material requires variable transparency");
  ASSERT(CONTAINS(layout->elements, "textureMask"),
         "material requires variable textureMask");

  // variable  types
  ASSERT(layout->elements["emission"].dtype == eFLOAT4,
         "material emission should be float4");
  ASSERT(layout->elements["baseColor"].dtype == eFLOAT4,
         "material baseColor should be float4");
  ASSERT(layout->elements["fresnel"].dtype == eFLOAT,
         "material fresnel should be float");
  ASSERT(layout->elements["roughness"].dtype == eFLOAT,
         "material roughness should be float");
  ASSERT(layout->elements["metallic"].dtype == eFLOAT,
         "material metallic should be float");
  ASSERT(layout->elements["transparency"].dtype == eFLOAT,
         "material transparency should be float");
  ASSERT(layout->elements["textureMask"].dtype == eINT,
         "material textureMask should be int");
}

void verifyObjectBuffer(std::shared_ptr<StructDataLayout> layout) {
  ASSERT(layout->elements.size() >= 2,
         "object buffer requires modelMatrix and segmentation");

  ASSERT(CONTAINS(layout->elements, "modelMatrix"),
         "object buffer requires variable modelMatrix");
  ASSERT(CONTAINS(layout->elements, "segmentation"),
         "object buffer requires variable modelMatrix");

  ASSERT(layout->elements["modelMatrix"].dtype == eFLOAT44,
         "object modelMatrix should be float44");
  ASSERT(layout->elements["segmentation"].dtype == eUINT4,
         "object segmentation should be uint4");
  ASSERT(!CONTAINS(layout->elements, "prevModelMatrix") ||
             layout->elements["prevModelMatrix"].dtype == eFLOAT44,
         "object prevModelMatrix should be float44");
  ASSERT(!CONTAINS(layout->elements, "userData") ||
             layout->elements["userData"].dtype == eFLOAT44,
         "object userData should be float44");
}

void verifySceneBuffer(std::shared_ptr<StructDataLayout> layout) {
  ASSERT(CONTAINS(layout->elements, "ambientLight"),
         "scene buffer requires variable ambientLight");
  ASSERT(CONTAINS(layout->elements, "directionalLights"),
         "scene buffer requires variable directionalLights");
  ASSERT(CONTAINS(layout->elements, "spotLights"),
         "scene buffer requires variable spotLights");
  ASSERT(CONTAINS(layout->elements, "pointLights"),
         "scene buffer requires variable pointLights");

  ASSERT(layout->elements["ambientLight"].dtype == eFLOAT4,
         "scene ambientLight should be float4");

  ASSERT(layout->elements["directionalLights"].dtype == eSTRUCT,
         "scene directionalLights should be struct");

  ASSERT(layout->elements["directionalLights"].member->elements.size() == 2 &&
             CONTAINS(layout->elements["directionalLights"].member->elements,
                      "direction") &&
             layout->elements["directionalLights"]
                     .member->elements["direction"]
                     .offset == 0 &&
             layout->elements["directionalLights"]
                     .member->elements["direction"]
                     .dtype == eFLOAT4 &&
             CONTAINS(layout->elements["directionalLights"].member->elements,
                      "emission") &&
             layout->elements["directionalLights"]
                     .member->elements["emission"]
                     .dtype == eFLOAT4,
         "directional lights in scene buffer must be an array of {vec4 "
         "direction; vec4 emission;}");

  ASSERT(layout->elements["spotLights"].dtype == eSTRUCT,
         "scene spotLights should be struct");

  ASSERT(
      layout->elements["spotLights"].member->elements.size() == 3 &&
          CONTAINS(layout->elements["spotLights"].member->elements,
                   "position") &&
          layout->elements["spotLights"].member->elements["position"].offset ==
              0 &&
          layout->elements["spotLights"].member->elements["position"].dtype ==
              eFLOAT4 &&
          CONTAINS(layout->elements["spotLights"].member->elements,
                   "direction") &&
          layout->elements["spotLights"].member->elements["direction"].offset ==
              16 &&
          layout->elements["spotLights"].member->elements["direction"].dtype ==
              eFLOAT4 &&
          CONTAINS(layout->elements["spotLights"].member->elements,
                   "emission") &&
          layout->elements["spotLights"].member->elements["emission"].dtype ==
              eFLOAT4,
      "spot lights in scene buffer must be an array of {vec4 "
      "direction; vec4 emission;}");

  ASSERT(
      layout->elements["pointLights"].member->elements.size() == 2 &&
          CONTAINS(layout->elements["pointLights"].member->elements,
                   "position") &&
          layout->elements["pointLights"].member->elements["position"].offset ==
              0 &&
          layout->elements["pointLights"].member->elements["position"].dtype ==
              eFLOAT4 &&
          CONTAINS(layout->elements["pointLights"].member->elements,
                   "emission") &&
          layout->elements["pointLights"].member->elements["emission"].dtype ==
              eFLOAT4,
      "point lights in scene buffer must be an array of {vec4 position; vec4 "
      "emission;}");

  ASSERT(!CONTAINS(layout->elements, "shadowMatrix") ||
             layout->elements["shadowMatrix"].dtype == eFLOAT44,
         "scene shadowMatrix should have type float44");
}

void verifyLightSpaceBuffer(std::shared_ptr<StructDataLayout> layout) {
  ASSERT(layout->elements.size() == 4, "Space buffer should contain the "
                                       "following elements: ViewMatrix, "
                                       "ProjectionMatrix, ViewMatrixInverse, "
                                       "ProjectionMatrixInverse");

  // required fields
  ASSERT(CONTAINS(layout->elements, "viewMatrix"),
         "camera buffer requires viewMatrix");
  ASSERT(CONTAINS(layout->elements, "projectionMatrix"),
         "camera buffer requires projectionMatrix");

  ASSERT(CONTAINS(layout->elements, "viewMatrixInverse"),
         "camera buffer requires viewMatrixInverse");
  ASSERT(CONTAINS(layout->elements, "projectionMatrixInverse"),
         "camera buffer requires projectionMatrixInverse");

  // required types
  ASSERT(layout->elements["viewMatrix"].dtype == eFLOAT44,
         "camera ViewMatrix should have type float44");
  ASSERT(layout->elements["projectionMatrix"].dtype == eFLOAT44,
         "camera ProjectionMatrix should have type float44");
  ASSERT(layout->elements["viewMatrixInverse"].dtype == eFLOAT44,
         "camera ViewMatrixInverse should have type float44");
  ASSERT(layout->elements["projectionMatrixInverse"].dtype == eFLOAT44,
         "camera ProjectionMatrixInverse should have type float44");
}

std::shared_ptr<StructDataLayout>
parseCameraBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                  uint32_t setNumber) {
  auto layout = parseBuffer(compiler, bindingNumber, setNumber);
  verifyCameraBuffer(layout);
  return layout;
}

std::shared_ptr<StructDataLayout>
parseMaterialBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                    uint32_t setNumber) {
  auto layout = parseBuffer(compiler, bindingNumber, setNumber);
  verifyMaterialBuffer(layout);
  return layout;
}

std::shared_ptr<StructDataLayout>
parseObjectBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                  uint32_t setNumber) {
  auto layout = parseBuffer(compiler, bindingNumber, setNumber);
  verifyObjectBuffer(layout);
  return layout;
}

std::shared_ptr<StructDataLayout>
parseSceneBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                 uint32_t setNumber) {
  auto layout = parseBuffer(compiler, bindingNumber, setNumber);
  verifySceneBuffer(layout);
  return layout;
}

std::shared_ptr<StructDataLayout>
parseLightSpaceBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                      uint32_t setNumber) {
  auto layout = parseBuffer(compiler, bindingNumber, setNumber);
  verifyLightSpaceBuffer(layout);
  return layout;
}

std::shared_ptr<SpecializationConstantLayout>
parseSpecializationConstant(spirv_cross::Compiler &compiler) {
  auto layout = std::make_shared<SpecializationConstantLayout>();
  auto constants = compiler.get_specialization_constants();
  for (auto &var : constants) {
    auto name = compiler.get_name(var.id);
    auto &constant = compiler.get_constant(var.id);
    auto type = compiler.get_type(constant.constant_type);
    auto dataType = get_data_type(type);
    layout->elements[name] = {
        .name = name,
        .id = var.constant_id,
        .dtype = dataType,
    };
    if (dataType == eINT) {
      layout->elements[name].intValue = constant.scalar_i32();
    } else if (dataType == eFLOAT) {
      layout->elements[name].intValue = constant.scalar_f32();
    } else {
      throw std::runtime_error(
          "only int and float are supported specialization constant types");
    }
  }
  return layout;
}

DescriptorSetDescription
getDescriptorSetDescription(spirv_cross::Compiler &compiler,
                            uint32_t setNumber) {
  DescriptorSetDescription result;
  auto resources = compiler.get_shader_resources();
  for (auto &r : resources.uniform_buffers) {
    if (compiler.get_decoration(
            r.id, spv::Decoration::DecorationDescriptorSet) == setNumber) {
      uint32_t bindingNumber =
          compiler.get_decoration(r.id, spv::Decoration::DecorationBinding);
      int dim = compiler.get_type(r.type_id).array.size();
      int arraySize = dim == 1 ? compiler.get_type(r.type_id).array[0] : 0;
      result.buffers.push_back(parseBuffer(compiler, r));
      result.bindings[bindingNumber] = {
          .name = r.name,
          .type = vk::DescriptorType::eUniformBuffer,
          .dim = dim,
          .arraySize = arraySize,
          .arrayIndex = static_cast<uint32_t>(result.buffers.size() - 1)};
    }
  }
  for (auto &r : resources.sampled_images) {
    if (compiler.get_decoration(
            r.id, spv::Decoration::DecorationDescriptorSet) == setNumber) {
      uint32_t bindingNumber =
          compiler.get_decoration(r.id, spv::Decoration::DecorationBinding);
      result.samplers.push_back(r.name);
      // compiler.get_type(r.type_id).image.dim == spv::DimCube);
      int dim = compiler.get_type(r.type_id).array.size();
      int arraySize = dim == 1 ? compiler.get_type(r.type_id).array[0] : 0;
      result.bindings[bindingNumber] = {
          .name = r.name,
          .type = vk::DescriptorType::eCombinedImageSampler,
          .dim = dim,
          .arraySize = arraySize,
          .arrayIndex = static_cast<uint32_t>(result.samplers.size() - 1)};
    }
  }

  if (result.bindings.empty()) {
    result.type = UniformBindingType::eNone;
    return result;
  }
  if (result.bindings.find(0) == result.bindings.end()) {
    throw std::runtime_error("All descriptor set must have binding=0, this is "
                             "used to identify its type.");
  }
  if (result.bindings[0].type == vk::DescriptorType::eCombinedImageSampler) {
    for (auto &b : result.bindings) {
      if (b.second.type != vk::DescriptorType::eCombinedImageSampler) {
        throw std::runtime_error("When a texture is bound at binding=0, all "
                                 "other bindings must be textures.");
      }
    }
    result.type = UniformBindingType::eTextures;
    return result;
  }
  if (result.bindings[0].type == vk::DescriptorType::eUniformBuffer) {
    auto name = result.bindings[0].name;
    if (name == "CameraBuffer") {
      result.type = UniformBindingType::eCamera;
      verifyCameraBuffer(result.buffers[result.bindings[0].arrayIndex]);
      return result;
    }
    if (name == "ObjectBuffer") {
      result.type = UniformBindingType::eObject;
      verifyObjectBuffer(result.buffers[result.bindings[0].arrayIndex]);
      return result;
    }
    if (name == "SceneBuffer") {
      result.type = UniformBindingType::eScene;
      verifySceneBuffer(result.buffers[result.bindings[0].arrayIndex]);
      return result;
    }
    if (name == "MaterialBuffer") {
      result.type = UniformBindingType::eMaterial;
      verifyMaterialBuffer(result.buffers[result.bindings[0].arrayIndex]);
      return result;
    }
    if (name == "LightBuffer") {
      result.type = UniformBindingType::eLight;
      verifyLightSpaceBuffer(result.buffers[result.bindings[0].arrayIndex]);
      return result;
    }
  }
  throw std::runtime_error(
      "Parse descriptor set failed: cannot recognize this set.");
}

std::future<void> BaseParser::loadGLSLFilesAsync(std::string const &vertFile,
                                                 std::string const &fragFile) {
  return std::async(std::launch::async,
                    [=, this]() { loadGLSLFiles(vertFile, fragFile); });
}

void BaseParser::loadGLSLFiles(std::string const &vertFile,
                               std::string const &fragFile) {
  log::info("Compiling: " + vertFile);
  mVertSPVCode = GLSLCompiler::compileGlslFileCached(
      vk::ShaderStageFlagBits::eVertex, vertFile);
  log::info("Compiled: " + vertFile);

  if (fragFile.length()) {
    log::info("Compiling: " + fragFile);
    mFragSPVCode = GLSLCompiler::compileGlslFileCached(
        vk::ShaderStageFlagBits::eFragment, fragFile);
    log::info("Compiled: " + fragFile);
  }

  try {
    reflectSPV();
  } catch (std::runtime_error const &err) {
    throw std::runtime_error("[" + vertFile + "|" + fragFile + "]" +
                             std::string(err.what()));
  }
}

void BaseParser::loadSPVFiles(std::string const &vertFile,
                              std::string const &fragFile) {
  std::vector<char> vertCodeRaw = readFile(vertFile);
  if (vertCodeRaw.size() / 4 * 4 != vertCodeRaw.size()) {
    throw std::runtime_error("corrupted SPV file: " + vertFile);
  }
  std::vector<uint32_t> vertCode(vertCodeRaw.size() / 4);
  std::memcpy(vertCode.data(), vertCodeRaw.data(), vertCodeRaw.size());

  std::vector<uint32_t> fragCode;
  if (fragFile.length()) {
    std::vector<char> fragCodeRaw = readFile(fragFile);
    if (fragCodeRaw.size() / 4 * 4 != fragCodeRaw.size()) {
      throw std::runtime_error("corrupted SPV file: " + fragFile);
    }
    fragCode = std::vector<uint32_t>(fragCodeRaw.size() / 4);
    std::memcpy(fragCode.data(), fragCodeRaw.data(), fragCodeRaw.size());
  }

  try {
    loadSPVCode(vertCode, fragCode);
  } catch (std::runtime_error const &err) {
    throw std::runtime_error(vertFile + "|" + fragFile +
                             std::string(err.what()));
  }
}

void BaseParser::loadSPVCode(std::vector<uint32_t> const &vertCode,
                             std::vector<uint32_t> const &fragCode) {
  mVertSPVCode = vertCode;
  mFragSPVCode = fragCode;
  reflectSPV();
}

std::vector<UniformBindingType> BaseParser::getUniformBindingTypes() const {
  return {};
}

std::vector<std::string> BaseParser::getInputTextureNames() const { return {}; }

} // namespace shader
} // namespace svulkan2
