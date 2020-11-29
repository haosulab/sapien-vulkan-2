#pragma once

#include <spirv_reflect.h>

namespace svulkan2 {

uint32_t GetSizeFromReflectFormat(SpvReflectFormat format);
std::string GetTypeNameFromReflectFormat(SpvReflectFormat format);

} // namespace svulkan2
