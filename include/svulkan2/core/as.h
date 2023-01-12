#pragma once
#include "buffer.h"
#include "command_pool.h"

namespace svulkan2 {
namespace core {

class BLAS {
  // public:
  //   static void BatchBuild(std::vector<BLAS *> const &blasArray);
public:
  BLAS(std::vector<vk::AccelerationStructureGeometryKHR> const &geometries,
       std::vector<vk::AccelerationStructureBuildRangeInfoKHR> const
           &buildRanges);

  void build(bool compaction = true);

  vk::DeviceAddress getAddress();

private:
  std::vector<vk::AccelerationStructureGeometryKHR> mGeometries;
  std::vector<vk::AccelerationStructureBuildRangeInfoKHR> mBuildRanges;

  std::unique_ptr<Buffer> mBuffer;
  vk::UniqueAccelerationStructureKHR mAS;
};

class TLAS {

public:
  TLAS(std::vector<vk::AccelerationStructureInstanceKHR> const &instances);
  void build();
  void update(std::vector<vk::TransformMatrixKHR> const &transforms);

  vk::DeviceAddress getAddress();

  vk::AccelerationStructureKHR getVulkanAS() const { return mAS.get(); };

private:
  std::vector<vk::AccelerationStructureInstanceKHR> mInstances;

  std::unique_ptr<CommandPool> mUpdateCommandPool;

  std::unique_ptr<Buffer> mInstanceBuffer;
  vk::DeviceAddress mInstanceBufferAddress;

  std::unique_ptr<Buffer> mUpdateScratchBuffer;
  vk::DeviceAddress mUpdateScratchBufferAddress;

  std::unique_ptr<Buffer> mBuffer;
  vk::UniqueAccelerationStructureKHR mAS;
};

} // namespace core
} // namespace svulkan2
