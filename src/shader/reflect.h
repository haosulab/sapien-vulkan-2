#pragma once

#include "svulkan2/common/layout.h"
#include <spirv_cross.hpp>

namespace svulkan2 {

bool type_is_int(spirv_cross::SPIRType const &type);
bool type_is_uint(spirv_cross::SPIRType const &type);
bool type_is_float(spirv_cross::SPIRType const &type);

bool type_is_int2(spirv_cross::SPIRType const &type);
bool type_is_uint2(spirv_cross::SPIRType const &type);
bool type_is_float2(spirv_cross::SPIRType const &type);

bool type_is_int3(spirv_cross::SPIRType const &type);
bool type_is_uint3(spirv_cross::SPIRType const &type);
bool type_is_float3(spirv_cross::SPIRType const &type);

bool type_is_int4(spirv_cross::SPIRType const &type);
bool type_is_uint4(spirv_cross::SPIRType const &type);
bool type_is_float4(spirv_cross::SPIRType const &type);

bool type_is_int44(spirv_cross::SPIRType const &type);
bool type_is_uint44(spirv_cross::SPIRType const &type);
bool type_is_float44(spirv_cross::SPIRType const &type);

bool type_is_struct(spirv_cross::SPIRType const &type);

DataType get_data_type(spirv_cross::SPIRType const &type);

spirv_cross::Resource *
find_uniform_by_decoration(spirv_cross::Compiler &compiler,
                           spirv_cross::ShaderResources &resource,
                           uint32_t binding_number, uint32_t set_number);

spirv_cross::Resource *
find_sampler_by_decoration(spirv_cross::Compiler &compiler,
                           spirv_cross::ShaderResources &resource,
                           uint32_t binding_number, uint32_t set_number);

vk::Format get_image_format(spv::ImageFormat format);
} // namespace svulkan2
