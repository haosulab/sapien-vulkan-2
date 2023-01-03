#pragma once
#include "buffer.h"

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

private:
  std::vector<vk::AccelerationStructureGeometryKHR> mGeometries;
  std::vector<vk::AccelerationStructureBuildRangeInfoKHR> mBuildRanges;

  std::unique_ptr<Buffer> mBuffer;
  vk::UniqueAccelerationStructureKHR mAS;
};

} // namespace core
} // namespace svulkan2
