#include "svulkan2/shader/reflect.h"
#include <stdexcept>

namespace svulkan2 {

uint32_t GetSizeFromReflectFormat(SpvReflectFormat format) {
  switch (format) {
  case SPV_REFLECT_FORMAT_R32_UINT:
    return 4;
  case SPV_REFLECT_FORMAT_R32_SINT:
    return 4;
  case SPV_REFLECT_FORMAT_R32_SFLOAT:
    return 4;
  case SPV_REFLECT_FORMAT_R32G32_UINT:
    return 8;
  case SPV_REFLECT_FORMAT_R32G32_SINT:
    return 8;
  case SPV_REFLECT_FORMAT_R32G32_SFLOAT:
    return 8;
  case SPV_REFLECT_FORMAT_R32G32B32_UINT:
    return 12;
  case SPV_REFLECT_FORMAT_R32G32B32_SINT:
    return 12;
  case SPV_REFLECT_FORMAT_R32G32B32_SFLOAT:
    return 12;
  case SPV_REFLECT_FORMAT_R32G32B32A32_UINT:
    return 16;
  case SPV_REFLECT_FORMAT_R32G32B32A32_SINT:
    return 16;
  case SPV_REFLECT_FORMAT_R32G32B32A32_SFLOAT:
    return 16;
  case SPV_REFLECT_FORMAT_UNDEFINED:
    return 0;
  }
  throw std::runtime_error("undefined SPV Reflect Format");
}

std::string GetTypeNameFromReflectFormat(SpvReflectFormat format) {
  switch (format) {
  case SPV_REFLECT_FORMAT_R32_UINT:
    return "uint";
  case SPV_REFLECT_FORMAT_R32_SINT:
    return "int";
  case SPV_REFLECT_FORMAT_R32_SFLOAT:
    return "float";
  case SPV_REFLECT_FORMAT_R32G32_UINT:
    return "uint2";
  case SPV_REFLECT_FORMAT_R32G32_SINT:
    return "int2";
  case SPV_REFLECT_FORMAT_R32G32_SFLOAT:
    return "float2";
  case SPV_REFLECT_FORMAT_R32G32B32_UINT:
    return "uint3";
  case SPV_REFLECT_FORMAT_R32G32B32_SINT:
    return "int3";
  case SPV_REFLECT_FORMAT_R32G32B32_SFLOAT:
    return "float3";
  case SPV_REFLECT_FORMAT_R32G32B32A32_UINT:
    return "uint4";
  case SPV_REFLECT_FORMAT_R32G32B32A32_SINT:
    return "int4";
  case SPV_REFLECT_FORMAT_R32G32B32A32_SFLOAT:
    return "float4";
  case SPV_REFLECT_FORMAT_UNDEFINED:
    return "undefined";
  }
  throw std::runtime_error("undefined SPV Reflect Format");
}

} // namespace svulkan2
