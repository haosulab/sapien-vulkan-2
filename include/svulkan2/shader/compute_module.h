#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace core {
class Context;
class Buffer;
class CommandPool;
class CommandBuffer;
} // namespace core

namespace shader {

class ComputeModule {
  static std::mutex SharedCommandPoolLock;
  static std::weak_ptr<core::CommandPool> SharedCommandPool;

public:
  ComputeModule(std::string const &filename, int blockSizeX, int blockSizeY = 1,
                int blockSizeZ = 1);
  ~ComputeModule();

  ComputeModule(ComputeModule const &) = delete;
  ComputeModule &operator=(ComputeModule const &) = delete;
  ComputeModule(ComputeModule const &&) = delete;
  ComputeModule &operator=(ComputeModule const &&) = delete;

  // TODO: make more general

  template <typename T> void bindConstantData(std::string const &name, T data) {
    auto &elem = mPushConstantLayout->elements.at(name);
    if (sizeof(T) != elem.size) {
      throw std::runtime_error("failed to bind constant data: size mismatch");
    }
    std::memcpy(mConstantBuffer.data() + elem.offset, &data, sizeof(T));
  }

  void bindConstantData(void *data, size_t size);
  void bindBuffers(std::vector<core::Buffer *> buffers);
  void record(core::CommandBuffer &cb, int x, int y, int z);

protected:
  void reflect();
  void compile();

  std::array<int, 3> mBlockSize;

  std::shared_ptr<core::Context> mContext;

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
