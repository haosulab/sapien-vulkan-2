#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace core {
class Buffer;
}

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
  inline std::shared_ptr<StructDataLayout> getPushConstantLayout() const {
    return mPushConstantLayout;
  }
  inline std::vector<uint32_t> const &getCode() const { return mSPVCode; }

  std::string summary() const;

private:
  std::vector<uint32_t> mSPVCode;
  std::unordered_map<uint32_t, DescriptorSetDescription> mResources;
  std::shared_ptr<StructDataLayout> mPushConstantLayout;
};

class RayTracingShaderPack {
public:
  RayTracingShaderPack(std::string const &shaderDir);

  std::future<void>
  loadGLSLFilesAsync(std::string const &raygenFile,
                     std::vector<std::string> const &missFiles,
                     std::vector<std::string> const &anyhitFiles,
                     std::vector<std::string> const &closestHitFiles);

  std::shared_ptr<InputDataLayout> computeCompatibleInputVertexLayout() const;

  inline std::unordered_map<uint32_t, DescriptorSetDescription> const &
  getResources() const {
    return mResources;
  }
  inline std::shared_ptr<StructDataLayout> getPushConstantLayout() const {
    return mPushConstantLayout;
  }

  std::string summary() const;

  inline RayTracingStageParser *getRaygenStageParser() const {
    return mRaygenStageParser.get();
  };
  inline std::vector<std::unique_ptr<RayTracingStageParser>> const &
  getMissStageParsers() const {
    return mMissStageParsers;
  }
  inline std::vector<std::unique_ptr<RayTracingStageParser>> const &
  getAnyHitStageParsers() const {
    return mAnyHitStageParsers;
  }
  inline std::vector<std::unique_ptr<RayTracingStageParser>> const &
  getClosestHitStageParsers() const {
    return mClosestHitStageParsers;
  }

private:
  std::unique_ptr<RayTracingStageParser> mRaygenStageParser;
  std::vector<std::unique_ptr<RayTracingStageParser>> mMissStageParsers;
  std::vector<std::unique_ptr<RayTracingStageParser>> mAnyHitStageParsers;
  std::vector<std::unique_ptr<RayTracingStageParser>> mClosestHitStageParsers;

  std::unordered_map<uint32_t, DescriptorSetDescription> mResources;
  std::shared_ptr<StructDataLayout> mPushConstantLayout;
};

class RayTracingShaderPackInstance {

public:
  struct RayTracingShaderPackInstanceDesc {
    std::string shaderDir{};
    uint32_t maxMeshes{};
    uint32_t maxMaterials{};
    uint32_t maxTextures{};
    uint32_t maxShapes{};
  };

  RayTracingShaderPackInstance(RayTracingShaderPackInstanceDesc desc);
  vk::Pipeline getPipeline();
  core::Buffer &getShaderBindingTable();

  RayTracingShaderPackInstance(RayTracingShaderPackInstance const &) = delete;
  RayTracingShaderPackInstance &
  operator=(RayTracingShaderPackInstance const &) = delete;

private:
  void initPipeline();
  void initSBT();

  RayTracingShaderPackInstanceDesc mDesc;

  vk::UniqueDescriptorSetLayout mSceneSetLayout;
  vk::UniqueDescriptorSetLayout mCameraSetLayout;
  vk::UniqueDescriptorSetLayout mOutputSetLayout;
  vk::UniquePipelineLayout mPipelineLayout;
  vk::UniquePipeline mPipeline;
  std::unique_ptr<core::Buffer> mSBTBuffer;

  std::shared_ptr<RayTracingShaderPack> mShaderPack;
};

} // namespace shader
} // namespace svulkan2
