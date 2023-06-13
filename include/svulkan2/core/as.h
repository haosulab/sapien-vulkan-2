#pragma once
#include "buffer.h"
#include "command_pool.h"

namespace svulkan2 {
namespace core {

class BLAS {
public:
  BLAS(std::vector<vk::AccelerationStructureGeometryKHR> const &geometries,
       std::vector<vk::AccelerationStructureBuildRangeInfoKHR> const &buildRanges,
       std::vector<uint32_t> const &maxPrimitiveCounts = {}, bool compaction = true,
       bool update = false);

  void build();
  void recordUpdate(vk::CommandBuffer commandBuffer,
                    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> const &buildRanges);

  vk::DeviceAddress getAddress();

private:
  std::vector<vk::AccelerationStructureGeometryKHR> mGeometries;
  std::vector<vk::AccelerationStructureBuildRangeInfoKHR> mBuildRanges;
  std::vector<uint32_t> mMaxPrimitiveCounts;
  bool mCompaction;
  bool mUpdate;

  std::unique_ptr<Buffer> mUpdateScratchBuffer;
  vk::DeviceAddress mUpdateScratchBufferAddress;

  std::unique_ptr<Buffer> mBuffer;
  vk::UniqueAccelerationStructureKHR mAS;
};

class TLAS {

public:
  TLAS(std::vector<vk::AccelerationStructureInstanceKHR> const &instances);
  void build();
  void recordUpdate(vk::CommandBuffer commandBuffer,
                    std::vector<vk::TransformMatrixKHR> const &transforms);

  vk::DeviceAddress getAddress();

  vk::AccelerationStructureKHR getVulkanAS() const { return mAS.get(); };

private:
  std::vector<vk::AccelerationStructureInstanceKHR> mInstances;

  std::unique_ptr<Buffer> mInstanceBuffer;
  vk::DeviceAddress mInstanceBufferAddress;

  std::unique_ptr<Buffer> mUpdateScratchBuffer;
  vk::DeviceAddress mUpdateScratchBufferAddress;

  std::unique_ptr<Buffer> mBuffer;
  vk::UniqueAccelerationStructureKHR mAS;
};

} // namespace core
} // namespace svulkan2
