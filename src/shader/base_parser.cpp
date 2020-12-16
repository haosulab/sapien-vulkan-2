#include "svulkan2/shader/base_parser.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {
namespace shader {

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

std::shared_ptr<StructDataLayout>
parseCameraBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                  uint32_t setNumber) {
  auto layout = parseBuffer(compiler, bindingNumber, setNumber);
  ASSERT(layout->elements.size() == 4 || layout->elements.size() == 6,
         "camera buffer should contain the following elements: viewMatrix, "
         "viewMatrixInverse, projectionMatrix, projectionMatrixInverse. If "
         "motion blur or optical flow is required, the following elements are "
         "supported: prevViewMatrix, prevViewMatrixInverse");

  // required fields
  ASSERT(CONTAINS(layout->elements, "viewMatrix"),
         "camera buffer requires viewMatrix");
  ASSERT(CONTAINS(layout->elements, "viewMatrixInverse"),
         "camera buffer requires viewMatrixInverse");
  ASSERT(CONTAINS(layout->elements, "projectionMatrix"),
         "camera buffer requires projectionMatrix");
  ASSERT(CONTAINS(layout->elements, "projectionMatrixInverse"),
         "camera buffer requires projectionMatrixInverse");
  if (layout->elements.size() == 6) {
    ASSERT(CONTAINS(layout->elements, "prevViewMatrix"),
           "camera buffer needs variable prevViewMatrix");
    ASSERT(CONTAINS(layout->elements, "prevViewMatrixInverse"),
           "camera buffer needs variable prevViewMatrixInverse");
  }

  // required types
  ASSERT(layout->elements["viewMatrix"].dtype == eFLOAT44,
         "camera viewMatrix should have type float44");
  ASSERT(layout->elements["viewMatrixInverse"].dtype == eFLOAT44,
         "camera viewMatrixInverse should have type float44");
  ASSERT(layout->elements["projectionMatrix"].dtype == eFLOAT44,
         "camera projectionMatrix should have type float44");
  ASSERT(layout->elements["projectionMatrixInverse"].dtype == eFLOAT44,
         "camera projectionMatrixInverse should have type float44");
  if (layout->elements.size() == 6) {
    ASSERT(layout->elements["prevViewMatrix"].dtype == eFLOAT44,
           "camera prevViewMatrix should have type float44");
    ASSERT(layout->elements["prevViewMatrixInverse"].dtype == eFLOAT44,
           "camera prevViewMatrixInverse should have type float44");
  }

  return layout;
}

std::shared_ptr<StructDataLayout>
parseMaterialBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                    uint32_t setNumber) {
  auto layout = parseBuffer(compiler, bindingNumber, setNumber);
  ASSERT(CONTAINS(layout->elements, "baseColor") ||
             CONTAINS(layout->elements, "diffuse"),
         "material buffer must contains the variable baseColor or diffuse");
  bool metallic = CONTAINS(layout->elements, "baseColor");
  if (metallic) {
    ASSERT(
        layout->elements.size() == 6,
        "metallic pipeline material should contain exactly 6 member variables");

    // required variables
    ASSERT(CONTAINS(layout->elements, "fresnel"),
           "metallic pipeline material requires variable fresnel");
    ASSERT(CONTAINS(layout->elements, "roughness"),
           "metallic pipeline material requires variable roughness");
    ASSERT(CONTAINS(layout->elements, "metallic"),
           "metallic pipeline material requires variable metallic");
    ASSERT(CONTAINS(layout->elements, "transparency"),
           "metallic pipeline material requires variable transparency");
    ASSERT(CONTAINS(layout->elements, "textureMask"),
           "metallic pipeline material requires variable textureMask");

    // variable  types
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
           "material textureMask should be float");
  } else {
    ASSERT(
        layout->elements.size() == 4,
        "specular pipeline material should contain exactly 4 member variables");

    // required variables
    ASSERT(CONTAINS(layout->elements, "specular"),
           "specular pipeline material requires variable roughness");
    ASSERT(CONTAINS(layout->elements, "transparency"),
           "specular pipeline material requires variable transparency");
    ASSERT(CONTAINS(layout->elements, "textureMask"),
           "specular pipeline material requires variable textureMask");

    // variable  types
    ASSERT(layout->elements["diffuse"].dtype == eFLOAT4,
           "material diffuse should be float4");
    ASSERT(layout->elements["specular"].dtype == eFLOAT4,
           "material specular should be float4");
    ASSERT(layout->elements["transparency"].dtype == eFLOAT,
           "material transparency should be float");
    ASSERT(layout->elements["textureMask"].dtype == eINT,
           "material textureMask should be float");
  }
  return layout;
}

std::shared_ptr<StructDataLayout>
parseObjectBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                  uint32_t setNumber) {
  auto layout = parseBuffer(compiler, bindingNumber, setNumber);

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

  return layout;
}

std::shared_ptr<StructDataLayout>
parseSceneBuffer(spirv_cross::Compiler &compiler, uint32_t bindingNumber,
                 uint32_t setNumber) {
  auto layout = parseBuffer(compiler, bindingNumber, setNumber);
  ASSERT(CONTAINS(layout->elements, "ambientLight"),
         "scene buffer requires variable ambientLight");
  ASSERT(CONTAINS(layout->elements, "directionalLights"),
         "scene buffer requires variable directionalLights");
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
  return layout;
}

std::shared_ptr<CombinedSamplerLayout>
parseCombinedSampler(spirv_cross::Compiler &compiler) {
  auto resources = compiler.get_shader_resources();
  auto samplers = resources.sampled_images;
  auto layout = std::make_shared<CombinedSamplerLayout>();

  for (auto &sampler : samplers) {
    uint32_t binding =
        compiler.get_decoration(sampler.id, spv::Decoration::DecorationBinding);
    uint32_t set = compiler.get_decoration(
        sampler.id, spv::Decoration::DecorationDescriptorSet);
    std::string name = sampler.name;
    layout->elements[name] = {.name = name, .binding = binding, .set = set};
  }
  return layout;
};

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

void BaseParser::loadGLSLFiles(std::string const &vertFile,
                               std::string const &fragFile) {
  GLSLCompiler compiler;
  mVertSPVCode = compiler.compileToSpirv(vk::ShaderStageFlagBits::eVertex,
                                         readFile(vertFile));
  log::info("shader compiled: " + vertFile);

  mFragSPVCode = compiler.compileToSpirv(vk::ShaderStageFlagBits::eFragment,
                                         readFile(fragFile));
  log::info("shader compiled: " + fragFile);

  try {
    reflectSPV();
  } catch (std::runtime_error const &err) {
    throw std::runtime_error(vertFile + "|" + fragFile +
                             std::string(err.what()));
  }
}

void BaseParser::loadSPVFiles(std::string const &vertFile,
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

} // namespace shader
} // namespace svulkan2
