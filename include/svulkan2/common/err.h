#pragma once
#include <stdexcept>
#include <string>

namespace svulkan2 {

inline void ASSERT(bool condition, std::string const &error) {
  if (!condition) {
    throw std::runtime_error(error);
  }
}

} // namespace svulkan2
