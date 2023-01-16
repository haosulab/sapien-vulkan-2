#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace shader {

class PostprocessingShaderParser {
public:
  std::future<void> loadFileAsync(std::string const &filepath);
  void reflectSPV();

  inline std::unordered_map<uint32_t, DescriptorSetDescription> const &
  getResources() const {
    return mResources;
  };

  inline std::vector<uint32_t> const &getCode() const { return mSPVCode; }

private:
  std::vector<uint32_t> mSPVCode;
  std::unordered_map<uint32_t, DescriptorSetDescription> mResources;
};

} // namespace shader
} // namespace svulkan2
