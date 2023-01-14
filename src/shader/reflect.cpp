#include "svulkan2/shader/reflect.h"
#include <stdexcept>

namespace svulkan2 {

bool type_is_int(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::Int && type.vecsize == 1 &&
         type.columns == 1;
}
bool type_is_uint(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::UInt && type.vecsize == 1 &&
         type.columns == 1;
}
bool type_is_float(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::Float && type.vecsize == 1 &&
         type.columns == 1;
}

bool type_is_int2(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::Int && type.vecsize == 2 &&
         type.columns == 1;
}
bool type_is_uint2(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::UInt && type.vecsize == 2 &&
         type.columns == 1;
}
bool type_is_float2(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::Float && type.vecsize == 2 &&
         type.columns == 1;
}

bool type_is_int3(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::Int && type.vecsize == 3 &&
         type.columns == 1;
}
bool type_is_uint3(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::UInt && type.vecsize == 3 &&
         type.columns == 1;
}
bool type_is_float3(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::Float && type.vecsize == 3 &&
         type.columns == 1;
}

bool type_is_int4(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::Int && type.vecsize == 4 &&
         type.columns == 1;
}
bool type_is_uint4(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::UInt && type.vecsize == 4 &&
         type.columns == 1;
}
bool type_is_float4(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::Float && type.vecsize == 4 &&
         type.columns == 1;
}

bool type_is_int44(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::Int && type.vecsize == 4 &&
         type.columns == 4;
}
bool type_is_uint44(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::UInt && type.vecsize == 4 &&
         type.columns == 4;
}
bool type_is_float44(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::Float && type.vecsize == 4 &&
         type.columns == 4;
}

bool type_is_struct(spirv_cross::SPIRType const &type) {
  return type.basetype == spirv_cross::SPIRType::Struct;
}

DataType get_data_type(spirv_cross::SPIRType const &type) {
  if (type_is_int(type)) {
    return DataType::eINT;
  }
  if (type_is_int2(type)) {
    return DataType::eINT2;
  }
  if (type_is_int3(type)) {
    return DataType::eINT3;
  }
  if (type_is_int4(type)) {
    return DataType::eINT4;
  }
  if (type_is_int44(type)) {
    return DataType::eINT44;
  }

  if (type_is_float(type)) {
    return DataType::eFLOAT;
  }
  if (type_is_float2(type)) {
    return DataType::eFLOAT2;
  }
  if (type_is_float3(type)) {
    return DataType::eFLOAT3;
  }
  if (type_is_float4(type)) {
    return DataType::eFLOAT4;
  }
  if (type_is_float44(type)) {
    return DataType::eFLOAT44;
  }

  if (type_is_uint(type)) {
    return DataType::eUINT;
  }
  if (type_is_uint2(type)) {
    return DataType::eUINT2;
  }
  if (type_is_uint3(type)) {
    return DataType::eUINT3;
  }
  if (type_is_uint4(type)) {
    return DataType::eUINT4;
  }
  if (type_is_uint44(type)) {
    return DataType::eUINT44;
  }

  if (type_is_struct(type)) {
    return DataType::eSTRUCT;
  }

  return DataType::eUNKNOWN;
}

spirv_cross::Resource *
find_uniform_by_decoration(spirv_cross::Compiler &compiler,
                           spirv_cross::ShaderResources &resource,
                           uint32_t binding_number, uint32_t set_number) {

  for (auto &r : resource.uniform_buffers) {
    if (compiler.get_decoration(r.id, spv::Decoration::DecorationBinding) ==
            binding_number &&
        compiler.get_decoration(
            r.id, spv::Decoration::DecorationDescriptorSet) == set_number) {
      return &r;
    }
  }
  return nullptr;
}

spirv_cross::Resource *
find_sampler_by_decoration(spirv_cross::Compiler &compiler,
                           spirv_cross::ShaderResources &resource,
                           uint32_t binding_number, uint32_t set_number) {

  for (auto &r : resource.sampled_images) {
    if (compiler.get_decoration(r.id, spv::Decoration::DecorationBinding) ==
            binding_number &&
        compiler.get_decoration(
            r.id, spv::Decoration::DecorationDescriptorSet) == set_number) {
      return &r;
    }
  }
  return nullptr;
}

vk::Format get_image_format(spv::ImageFormat format) {
  switch (format) {
  case spv::ImageFormat::ImageFormatUnknown:
    return vk::Format::eUndefined;

  case spv::ImageFormat::ImageFormatRgba32f:
    return vk::Format::eR32G32B32A32Sfloat;
  case spv::ImageFormat::ImageFormatR32f:
    return vk::Format::eR32Sfloat;

  case spv::ImageFormat::ImageFormatRgba16f:
    return vk::Format::eR16G16B16A16Sfloat;
  case spv::ImageFormat::ImageFormatR16f:
    return vk::Format::eR16Sfloat;

  case spv::ImageFormat::ImageFormatRgba32i:
    return vk::Format::eR32G32B32A32Sint;
  case spv::ImageFormat::ImageFormatRgba32ui:
    return vk::Format::eR32G32B32A32Uint;

  case spv::ImageFormat::ImageFormatRgba8:
    return vk::Format::eR8G8B8A8Unorm;
  case spv::ImageFormat::ImageFormatR8:
    return vk::Format::eR8Unorm;

  default:
    throw std::runtime_error("Unsupported spirv format.");
  }
}

} // namespace svulkan2
