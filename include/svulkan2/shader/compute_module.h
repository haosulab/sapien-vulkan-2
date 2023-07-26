#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace core {
class Buffer;
}

namespace shader {

class ComputeModule {
public:
  ComputeModule(std::string const &filename);
  ~ComputeModule();

  ComputeModule(ComputeModule const &) = delete;
  ComputeModule &operator=(ComputeModule const &) = delete;
  ComputeModule(ComputeModule const &&) = delete;
  ComputeModule &operator=(ComputeModule const &&) = delete;

  // TODO: make more general
  void bindConstantData(void *data, size_t size);
  void bindBuffers(std::vector<core::Buffer *> buffers);
  void record(vk::CommandBuffer cb, int x, int y = 1, int z = 1);

protected:
  void reflect();
  void compile();

  std::vector<uint32_t> mCode;
  std::vector<DescriptorSetDescription> mDescriptorSetDescriptions;
  std::shared_ptr<SpecializationConstantLayout> mSpecializationConstantLayout;
  std::shared_ptr<StructDataLayout> mPushConstantLayout;

  vk::UniquePipelineLayout mPipelineLayout;
  vk::UniquePipeline mPipeline;
  std::vector<vk::UniqueDescriptorSetLayout> mSetLayouts;
  std::vector<vk::UniqueDescriptorSet> mSets;

  std::vector<char> mConstantBuffer;
};

} // namespace shader
} // namespace svulkan2
