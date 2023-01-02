#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace shader {

class RayTracingStageParser {
public:
  std::future<void> loadFileAsync(std::string const &filepath,
                                  vk::ShaderStageFlagBits stage);
  void reflectSPV();
  inline std::unordered_map<uint32_t, DescriptorSetDescription> const &
  getResources() const {
    return mResources;
  }

  std::string summary() const;

private:
  std::vector<uint32_t> mSPVCode;
  std::unordered_map<uint32_t, DescriptorSetDescription> mResources;
};

class RayTracingParser {
public:
  std::future<void>
  loadGLSLFilesAsync(std::string const &raygenFile,
                     std::vector<std::string> const &missFiles,
                     std::vector<std::string> const &anyhitFiles,
                     std::vector<std::string> const &closestHitFiles);

  std::string summary() const;

private:
  std::unique_ptr<RayTracingStageParser> mRaygenStageParser;
  std::vector<std::unique_ptr<RayTracingStageParser>> mMissStageParsers;
  std::vector<std::unique_ptr<RayTracingStageParser>> mAnyHitStageParsers;
  std::vector<std::unique_ptr<RayTracingStageParser>> mClosestHitStageParsers;

  std::unordered_map<uint32_t, DescriptorSetDescription> mResources;
};

} // namespace shader
} // namespace svulkan2
