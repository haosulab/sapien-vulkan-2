#include "svulkan2/shader/base_parser.h"
#include "reflect.h"
#include "svulkan2/common/launch_policy.h"
#include "../common/logger.h"

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
  std::vector<std::string> newImages = images;
  std::map<uint32_t, Binding> newBindings = bindings;

  for (auto &[bindingIndex, binding] : other.bindings) {
    if (bindings.find(bindingIndex) != bindings.end()) {
      if (bindings.at(bindingIndex).type != binding.type) {
        throw std::runtime_error("Incompatible descriptor at binding " +
                                 std::to_string(bindingIndex) +
                                 ": data types are different.");
      }

      switch (binding.type) {
      case vk::DescriptorType::eUniformBuffer:
      case vk::DescriptorType::eStorageBuffer:
        if (*other.buffers.at(binding.arrayIndex) !=
            *buffers.at(bindings.at(bindingIndex).arrayIndex)) {
          throw std::runtime_error("Incompatible descriptor at binding " +
                                   std::to_string(bindingIndex) +
                                   ": buffers are different.");
        }
        break;
      case vk::DescriptorType::eCombinedImageSampler:
        if (other.samplers.at(binding.arrayIndex) !=
            samplers[(bindings.at(bindingIndex)).arrayIndex]) {
          throw std::runtime_error("Incompatible descriptor at binding " +
                                   std::to_string(bindingIndex) +
                                   ": texture names are different.");
        }
        // TODO: check array size and imageDim as well
        break;
      case vk::DescriptorType::eStorageImage:
        if (other.images.at(binding.arrayIndex) !=
                samplers.at((bindings.at(bindingIndex)).arrayIndex) ||
            bindings.at(bindingIndex).format != binding.format) {
          throw std::runtime_error("Incompatible descriptor at binding " +
                                   std::to_string(bindingIndex) +
                                   ": image names are different.");
        }
        break;
      case vk::DescriptorType::eAccelerationStructureKHR:
        break;
      default:
        throw std::runtime_error(
            "Descriptor merge failed: unsupported descriptor type");
      }

    } else {

      switch (binding.type) {
      case vk::DescriptorType::eUniformBuffer:
      case vk::DescriptorType::eStorageBuffer:
        newBuffers.push_back(other.buffers[binding.arrayIndex]);
        newBindings[bindingIndex] = {
            .name = binding.name,
            .type = binding.type,
            .dim = binding.dim,
            .arraySize = binding.arraySize,
            .arrayIndex = static_cast<uint32_t>(newBuffers.size() - 1)};
        break;
      case vk::DescriptorType::eCombinedImageSampler:
        newSamplers.push_back(other.samplers[binding.arrayIndex]);
        newBindings[bindingIndex] = {
            .name = binding.name,
            .type = vk::DescriptorType::eCombinedImageSampler,
            .dim = binding.dim,
            .arraySize = binding.arraySize,
            .imageDim = binding.imageDim,
            .arrayIndex = static_cast<uint32_t>(newSamplers.size() - 1)};
        break;
      case vk::DescriptorType::eStorageImage:
        newImages.push_back(other.images[binding.arrayIndex]);
        newBindings[bindingIndex] = {
            .name = binding.name,
            .type = vk::DescriptorType::eStorageImage,
            .dim = binding.dim,
            .arraySize = binding.arraySize,
            .imageDim = binding.imageDim,
            .arrayIndex = static_cast<uint32_t>(newSamplers.size() - 1),
            .format = binding.format};
        break;
      case vk::DescriptorType::eAccelerationStructureKHR:
        newBindings[bindingIndex] = binding;
        break;
      default:
        throw std::runtime_error(
            "Descriptor merge failed: unsupported descriptor type");
      }
    }
  }
  newLayout.buffers = newBuffers;
  newLayout.samplers = newSamplers;
  newLayout.images = newImages;
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
  ASSERT(vertexData->elements.contains("position"),
         "vertex position is required at location 0 with type float3");

  // required type
  ASSERT(vertexData->elements["position"].dtype == DataType::eFLOAT3,
         "The following is required: layout(location = 0) in vec3 position;");
  ASSERT(!vertexData->elements.contains("normal") ||
             vertexData->elements["normal"].dtype == DataType::eFLOAT3,
         "normal in vertex input must be a float3");
  ASSERT(!vertexData->elements.contains("uv") ||
             vertexData->elements["uv"].dtype == DataType::eFLOAT2,
         "uv in vertex input must be a float2");
  ASSERT(!vertexData->elements.contains("tangent") ||
             vertexData->elements["tangent"].dtype == DataType::eFLOAT3,
         "tangent in vertex input must be a float3");
  ASSERT(!vertexData->elements.contains("bitangent") ||
             vertexData->elements["bitangent"].dtype == DataType::eFLOAT3,
         "bitangent in vertex input must be a float3");
  ASSERT(!vertexData->elements.contains("color") ||
             vertexData->elements["color"].dtype == DataType::eFLOAT4,
         "color in vertex input must be a float4");

  // required associated types
  ASSERT(!vertexData->elements.contains("tangent") ||
             vertexData->elements.contains("normal"),
         "normal in vertex input is required when using tangent");
  ASSERT(!vertexData->elements.contains("bitangent") ||
             (vertexData->elements.contains("normal") &&
              vertexData->elements.contains("tangent")),
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

  ASSERT(!textureOutput->elements.contains("outAlbedo") ||
             textureOutput->elements["outAlbedo"].dtype == DataType::eFLOAT4,
         "outAlbedo must be float4");
  ASSERT(!textureOutput->elements.contains("outPosition") ||
             textureOutput->elements["outPosition"].dtype == DataType::eFLOAT4,
         "outPosition must be float4");
  ASSERT(!textureOutput->elements.contains("outNormal") ||
             textureOutput->elements["outNormal"].dtype == DataType::eFLOAT4,
         "outNormal must be float4");
  ASSERT(!textureOutput->elements.contains("outSegmentation") ||
             textureOutput->elements["outSegmentation"].dtype ==
                 DataType::eUINT4,
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

    if (dataType == DataType::eSTRUCT) {
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
  ASSERT(layout->elements.contains("viewMatrix"),
         "camera buffer requires viewMatrix");
  ASSERT(layout->elements.contains("viewMatrixInverse"),
         "camera buffer requires viewMatrixInverse");
  ASSERT(layout->elements.contains("projectionMatrix"),
         "camera buffer requires projectionMatrix");
  ASSERT(layout->elements.contains("projectionMatrixInverse"),
         "camera buffer requires projectionMatrixInverse");

  // required types
  ASSERT(layout->elements["viewMatrix"].dtype == DataType::eFLOAT44,
         "camera viewMatrix should have type float44");
  ASSERT(layout->elements["viewMatrixInverse"].dtype == DataType::eFLOAT44,
         "camera viewMatrixInverse should have type float44");
  ASSERT(layout->elements["projectionMatrix"].dtype == DataType::eFLOAT44,
         "camera projectionMatrix should have type float44");
  ASSERT(layout->elements["projectionMatrixInverse"].dtype ==
             DataType::eFLOAT44,
         "camera projectionMatrixInverse should have type float44");
  if (layout->elements.contains("prevViewMatrix")) {
    ASSERT(layout->elements["prevViewMatrix"].dtype == DataType::eFLOAT44,
           "camera prevViewMatrix should have type float44");
  }
  if (layout->elements.contains("prevViewMatrixInverse")) {
    ASSERT(layout->elements["prevViewMatrixInverse"].dtype ==
               DataType::eFLOAT44,
           "camera prevViewMatrixInverse should have type float44");
  }
  if (layout->elements.contains("width")) {
    ASSERT(layout->elements["width"].dtype == DataType::eFLOAT,
           "camera width should have type float");
  }
  if (layout->elements.contains("height")) {
    ASSERT(layout->elements["height"].dtype == DataType::eFLOAT,
           "camera height should have type float");
  }
}

void verifyMaterialBuffer(std::shared_ptr<StructDataLayout> layout) {
  ASSERT(layout->size == 64, "Material should be exactly 64 bytes in size");

  // required variables
  ASSERT(layout->elements.contains("emission"),
         "material requires variable emission");
  ASSERT(layout->elements.contains("baseColor"),
         "material requires variable baseColor");
  ASSERT(layout->elements.contains("fresnel"),
         "material requires variable fresnel");
  ASSERT(layout->elements.contains("roughness"),
         "material requires variable roughness");
  ASSERT(layout->elements.contains("metallic"),
         "material requires variable metallic");
  ASSERT(layout->elements.contains("transmission"),
         "material requires variable transmission");
  ASSERT(layout->elements.contains("ior"), "material requires variable ior");
  ASSERT(layout->elements.contains("transmissionRoughness"),
         "material requires variable transmissionRoughness");
  ASSERT(layout->elements.contains("textureMask"),
         "material requires variable textureMask");

  // variable  types
  ASSERT(layout->elements["emission"].dtype == DataType::eFLOAT4,
         "material emission should be float4");
  ASSERT(layout->elements["baseColor"].dtype == DataType::eFLOAT4,
         "material baseColor should be float4");
  ASSERT(layout->elements["fresnel"].dtype == DataType::eFLOAT,
         "material fresnel should be float");
  ASSERT(layout->elements["roughness"].dtype == DataType::eFLOAT,
         "material roughness should be float");
  ASSERT(layout->elements["metallic"].dtype == DataType::eFLOAT,
         "material metallic should be float");
  ASSERT(layout->elements["transparency"].dtype == DataType::eFLOAT,
         "material transparency should be float");
  ASSERT(layout->elements["ior"].dtype == DataType::eFLOAT,
         "material ior should be float");
  ASSERT(layout->elements["transmissionRoughness"].dtype == DataType::eFLOAT,
         "material transmissionRoughness should be float");
  ASSERT(layout->elements["textureMask"].dtype == DataType::eINT,
         "material textureMask should be int");
}

void verifyObjectBuffer(std::shared_ptr<StructDataLayout> layout) {
  ASSERT(layout->elements.size() >= 2,
         "object buffer requires modelMatrix and segmentation");

  ASSERT(layout->elements.contains("modelMatrix"),
         "object buffer requires variable modelMatrix");
  ASSERT(layout->elements.contains("segmentation"),
         "object buffer requires variable modelMatrix");

  ASSERT(layout->elements["modelMatrix"].dtype == DataType::eFLOAT44,
         "object modelMatrix should be float44");
  ASSERT(layout->elements["segmentation"].dtype == DataType::eUINT4,
         "object segmentation should be uint4");
  ASSERT(!layout->elements.contains("prevModelMatrix") ||
             layout->elements["prevModelMatrix"].dtype == DataType::eFLOAT44,
         "object prevModelMatrix should be float44");
  ASSERT(!layout->elements.contains("userData") ||
             layout->elements["userData"].dtype == DataType::eFLOAT44,
         "object userData should be float44");
}

void verifySceneBuffer(std::shared_ptr<StructDataLayout> layout) {
  ASSERT(layout->elements.contains("ambientLight"),
         "scene buffer requires variable ambientLight");
  ASSERT(layout->elements.contains("directionalLights"),
         "scene buffer requires variable directionalLights");
  ASSERT(layout->elements.contains("spotLights"),
         "scene buffer requires variable spotLights");
  ASSERT(layout->elements.contains("pointLights"),
         "scene buffer requires variable pointLights");

  ASSERT(layout->elements["ambientLight"].dtype == DataType::eFLOAT4,
         "scene ambientLight should be float4");

  ASSERT(layout->elements["directionalLights"].dtype == DataType::eSTRUCT,
         "scene directionalLights should be struct");

  ASSERT(layout->elements["directionalLights"].member->elements.size() == 2 &&
             layout->elements["directionalLights"].member->elements.contains(
                 "direction") &&
             layout->elements["directionalLights"]
                     .member->elements["direction"]
                     .offset == 0 &&
             layout->elements["directionalLights"]
                     .member->elements["direction"]
                     .dtype == DataType::eFLOAT4 &&
             layout->elements["directionalLights"].member->elements.contains(
                 "emission") &&
             layout->elements["directionalLights"]
                     .member->elements["emission"]
                     .dtype == DataType::eFLOAT4,
         "directional lights in scene buffer must be an array of {vec4 "
         "direction; vec4 emission;}");

  ASSERT(layout->elements["spotLights"].dtype == DataType::eSTRUCT,
         "scene spotLights should be struct");

  ASSERT(
      layout->elements["spotLights"].member->elements.size() == 3 &&
          layout->elements["spotLights"].member->elements.contains(
              "position") &&
          layout->elements["spotLights"].member->elements["position"].offset ==
              0 &&
          layout->elements["spotLights"].member->elements["position"].dtype ==
              DataType::eFLOAT4 &&
          layout->elements["spotLights"].member->elements.contains(
              "direction") &&
          layout->elements["spotLights"].member->elements["direction"].offset ==
              16 &&
          layout->elements["spotLights"].member->elements["direction"].dtype ==
              DataType::eFLOAT4 &&
          layout->elements["spotLights"].member->elements.contains(
              "emission") &&
          layout->elements["spotLights"].member->elements["emission"].dtype ==
              DataType::eFLOAT4,
      "spot lights in scene buffer must be an array of {vec4 "
      "direction; vec4 emission;}");

  ASSERT(
      layout->elements["pointLights"].member->elements.size() == 2 &&
          layout->elements["pointLights"].member->elements.contains(
              "position") &&
          layout->elements["pointLights"].member->elements["position"].offset ==
              0 &&
          layout->elements["pointLights"].member->elements["position"].dtype ==
              DataType::eFLOAT4 &&
          layout->elements["pointLights"].member->elements.contains(
              "emission") &&
          layout->elements["pointLights"].member->elements["emission"].dtype ==
              DataType::eFLOAT4,
      "point lights in scene buffer must be an array of {vec4 position; vec4 "
      "emission;}");

  ASSERT(!layout->elements.contains("shadowMatrix") ||
             layout->elements["shadowMatrix"].dtype == DataType::eFLOAT44,
         "scene shadowMatrix should have type float44");
}

void verifyLightSpaceBuffer(std::shared_ptr<StructDataLayout> layout) {
  ASSERT(layout->elements.size() == 6,
         "light buffer should contain the "
         "following elements: viewMatrix, "
         "projectionMatrix, viewMatrixInverse, "
         "projectionMatrixInverse, width, height");

  // required fields
  ASSERT(layout->elements.contains("viewMatrix"),
         "light buffer requires viewMatrix");
  ASSERT(layout->elements.contains("projectionMatrix"),
         "light buffer requires projectionMatrix");
  ASSERT(layout->elements.contains("viewMatrixInverse"),
         "light buffer requires viewMatrixInverse");
  ASSERT(layout->elements.contains("projectionMatrixInverse"),
         "light buffer requires projectionMatrixInverse");
  ASSERT(layout->elements.contains("width"), "light buffer requires width");
  ASSERT(layout->elements.contains("height"), "light buffer requires height");

  // required types
  ASSERT(layout->elements["viewMatrix"].dtype == DataType::eFLOAT44,
         "light ViewMatrix should have type float44");
  ASSERT(layout->elements["projectionMatrix"].dtype == DataType::eFLOAT44,
         "light ProjectionMatrix should have type float44");
  ASSERT(layout->elements["viewMatrixInverse"].dtype == DataType::eFLOAT44,
         "light ViewMatrixInverse should have type float44");
  ASSERT(layout->elements["projectionMatrixInverse"].dtype ==
             DataType::eFLOAT44,
         "light ProjectionMatrixInverse should have type float44");
  ASSERT(layout->elements["width"].dtype == DataType::eINT,
         "light width should have type int");
  ASSERT(layout->elements["height"].dtype == DataType::eINT,
         "light height should have type int");
}

std::shared_ptr<SpecializationConstantLayout>
parseSpecializationConstant(spirv_cross::Compiler &compiler) {
  auto layout = std::make_shared<SpecializationConstantLayout>();
  auto constants = compiler.get_specialization_constants();
  for (auto &var : constants) {
    auto name = compiler.get_name(var.id);
    if (name.empty()) {
      continue;
      // printf("empty constant name???\n");
      // exit(1);
    }
    auto &constant = compiler.get_constant(var.id);
    auto type = compiler.get_type(constant.constant_type);
    auto dataType = get_data_type(type);
    layout->elements[name] = {
        .name = name,
        .id = var.constant_id,
        .dtype = dataType,
    };
    if (dataType == DataType::eINT) {
      layout->elements[name].intValue = constant.scalar_i32();
    } else if (dataType == DataType::eFLOAT) {
      layout->elements[name].floatValue = constant.scalar_f32();
    } else if (dataType == DataType::eUINT) {
      layout->elements[name].intValue = constant.scalar_i32();  // HACK
    } else {
      throw std::runtime_error(
          "only int and float are supported specialization constant types");
    }
  }
  return layout;
}

std::vector<uint32_t> getDescriptorSetIds(spirv_cross::Compiler &compiler) {
  auto resources = compiler.get_shader_resources();
  std::unordered_set<uint32_t> ids;
  for (auto &r : resources.uniform_buffers) {
    ids.insert(compiler.get_decoration(r.id, spv::Decoration::DecorationDescriptorSet));
  }
  for (auto &r : resources.sampled_images) {
    ids.insert(compiler.get_decoration(r.id, spv::Decoration::DecorationDescriptorSet));
  }
  for (auto &r : resources.storage_buffers) {
    ids.insert(compiler.get_decoration(r.id, spv::Decoration::DecorationDescriptorSet));
  }
  for (auto &r : resources.storage_images) {
    ids.insert(compiler.get_decoration(r.id, spv::Decoration::DecorationDescriptorSet));
  }
  std::vector<uint32_t> result = {ids.begin(), ids.end()};
  std::sort(result.begin(), result.end());
  return result;
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
      int imageDim = -1;
      if (compiler.get_type(r.type_id).image.dim == spv::Dim1D) {
        imageDim = 1;
      } else if (compiler.get_type(r.type_id).image.dim == spv::Dim2D) {
        imageDim = 2;
      } else if (compiler.get_type(r.type_id).image.dim == spv::Dim3D) {
        imageDim = 3;
      } else if (compiler.get_type(r.type_id).image.dim == spv::DimCube) {
        imageDim = 4;
      }
      int arraySize = dim == 1 ? compiler.get_type(r.type_id).array[0] : 0;
      result.bindings[bindingNumber] = {
          .name = r.name,
          .type = vk::DescriptorType::eCombinedImageSampler,
          .dim = dim,
          .arraySize = arraySize,
          .imageDim = imageDim,
          .arrayIndex = static_cast<uint32_t>(result.samplers.size() - 1)};
    }
  }
  for (auto &r: resources.storage_buffers) {
    if (compiler.get_decoration(
            r.id, spv::Decoration::DecorationDescriptorSet) == setNumber) {
      uint32_t bindingNumber =
          compiler.get_decoration(r.id, spv::Decoration::DecorationBinding);
      int dim = compiler.get_type(r.type_id).array.size();
      int arraySize = dim == 1 ? compiler.get_type(r.type_id).array[0] : 0;
      result.buffers.push_back(parseBuffer(compiler, r));
      result.bindings[bindingNumber] = {
          .name = r.name,
          .type = vk::DescriptorType::eStorageBuffer,
          .dim = dim,
          .arraySize = arraySize,
          .arrayIndex = static_cast<uint32_t>(result.buffers.size() - 1)};
    }
  }
  for (auto &r: resources.storage_images) {
    if (compiler.get_decoration(
            r.id, spv::Decoration::DecorationDescriptorSet) == setNumber) {
      uint32_t bindingNumber =
          compiler.get_decoration(r.id, spv::Decoration::DecorationBinding);
      result.images.push_back(r.name);

      int dim = compiler.get_type(r.type_id).array.size();
      int imageDim = -1;
      if (compiler.get_type(r.type_id).image.dim == spv::Dim1D) {
        imageDim = 1;
      } else if (compiler.get_type(r.type_id).image.dim == spv::Dim2D) {
        imageDim = 2;
      } else if (compiler.get_type(r.type_id).image.dim == spv::Dim3D) {
        imageDim = 3;
      } else if (compiler.get_type(r.type_id).image.dim == spv::DimCube) {
        imageDim = 4;
      }
      int arraySize = dim == 1 ? compiler.get_type(r.type_id).array[0] : 0;
      result.bindings[bindingNumber] = {
          .name = r.name,
          .type = vk::DescriptorType::eStorageImage,
          .dim = dim,
          .arraySize = arraySize,
          .imageDim = imageDim,
          .arrayIndex = static_cast<uint32_t>(result.images.size() - 1)};
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

  result.type = UniformBindingType::eUnknown;  // general compute shader binding
  // throw std::runtime_error("Parse descriptor set failed: cannot recognize this set.");
  return result;
}

std::future<void> BaseParser::loadGLSLFilesAsync(std::string const &vertFile,
                                                 std::string const &fragFile,
                                                 std::string const &geomFile) {
  return std::async(LAUNCH_ASYNC, [=, this]() {
    loadGLSLFiles(vertFile, fragFile, geomFile);
  });
}

void BaseParser::loadGLSLFiles(std::string const &vertFile,
                               std::string const &fragFile,
                               std::string const &geomFile) {
  logger::info("Compiling: " + vertFile);
  mVertSPVCode = GLSLCompiler::compileGlslFileCached(
      vk::ShaderStageFlagBits::eVertex, vertFile);
  logger::info("Compiled: " + vertFile);

  if (fragFile.length()) {
    logger::info("Compiling: " + fragFile);
    mFragSPVCode = GLSLCompiler::compileGlslFileCached(
        vk::ShaderStageFlagBits::eFragment, fragFile);
    logger::info("Compiled: " + fragFile);
  }

  if (geomFile.length()) {
    logger::info("Compiling: " + geomFile);
    mGeomSPVCode = GLSLCompiler::compileGlslFileCached(
        vk::ShaderStageFlagBits::eGeometry, geomFile);
    logger::info("Compiled: " + geomFile);
  }

  try {
    reflectSPV();
  } catch (std::runtime_error const &err) {
    throw std::runtime_error("[" + vertFile + "|" + fragFile + "]" +
                             std::string(err.what()));
  }
}

std::vector<UniformBindingType> BaseParser::getUniformBindingTypes() const {
  return {};
}

std::vector<std::string> BaseParser::getInputTextureNames() const { return {}; }

vk::UniquePipelineLayout BaseParser::createPipelineLayout(
    vk::Device device, std::vector<vk::DescriptorSetLayout> layouts) const {
  return device.createPipelineLayoutUnique(
      vk::PipelineLayoutCreateInfo({}, layouts));
}

} // namespace shader
} // namespace svulkan2
