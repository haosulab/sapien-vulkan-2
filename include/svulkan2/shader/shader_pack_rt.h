#pragma once

#include <memory>

namespace svulkan2 {
namespace core {
class Context;

struct ShaderPackRtDescription {
  std::string dirname;
  inline bool operator==(ShaderPackRtDescription const &other) const {
    return dirname == other.dirname;
  }
};

class ShaderPackRt {
public:
  ShaderPackRt(std::string const &dirname);
private:
};

} // namespace core
} // namespace svulkan2
