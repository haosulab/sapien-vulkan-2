#pragma once
#include <stdexcept>
#include <string>

namespace svulkan2 {

template <typename Container, typename T>
inline bool CONTAINS(Container &container, T const &element) {
  return container.find(element) != container.end();
}

inline void ASSERT(bool condition, std::string const &error) {
  if (!condition) {
    throw std::runtime_error(error);
  }
}

} // namespace svulkan2
