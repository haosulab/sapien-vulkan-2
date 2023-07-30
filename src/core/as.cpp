#include "svulkan2/core/as.h"
#include "../common/logger.h"
#include "svulkan2/core/command_buffer.h"
#include "svulkan2/core/command_pool.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace core {

BLAS::BLAS(std::vector<vk::AccelerationStructureGeometryKHR> const &geometries,
           std::vector<vk::AccelerationStructureBuildRangeInfoKHR> const &buildRanges,
           std::vector<uint32_t> const &maxPrimitiveCounts, bool compaction, bool update)
    : mGeometries(geometries), mBuildRanges(buildRanges), mCompaction(compaction),
      mUpdate(update) {
  if (compaction && update) {
    logger::error(
        "Current implementation does not allow compaction for dynamic acceleration structures.");
    mCompaction = false;
  }

  if (maxPrimitiveCounts.empty()) {
    for (auto &range : mBuildRanges) {
      mMaxPrimitiveCounts.push_back(range.primitiveCount);
    }
  } else {
    // sanity checks
    if (maxPrimitiveCounts.size() != buildRanges.size()) {
      throw std::runtime_error("buildRanges and maxPrimitiveCounts must have equal size");
    }
    for (size_t i = 0; i < maxPrimitiveCounts.size(); ++i) {
      if (maxPrimitiveCounts[i] < buildRanges[i].primitiveCount) {
        throw std::runtime_error(
            "maxPrimitiveCounts must be greater or equal to primitive count in buildRanges");
      }
    }
    mMaxPrimitiveCounts = maxPrimitiveCounts;
  }
}

void BLAS::build() {
  auto context = Context::Get();

  vk::BuildAccelerationStructureFlagsKHR flags{};
  if (mCompaction) {
    flags |= vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction;
  }
  if (mUpdate) {
    flags |= vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
  }

  vk::AccelerationStructureBuildGeometryInfoKHR buildInfo(
      vk::AccelerationStructureTypeKHR::eBottomLevel, flags,
      vk::BuildAccelerationStructureModeKHR::eBuild, nullptr, vk::AccelerationStructureKHR{},
      mGeometries, nullptr, vk::DeviceOrHostAddressKHR{});
  vk::AccelerationStructureBuildSizesInfoKHR sizeInfo =
      context->getDevice().getAccelerationStructureBuildSizesKHR(
          vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, mMaxPrimitiveCounts);
  vk::DeviceSize size = sizeInfo.accelerationStructureSize;

  auto asBuffer =
      std::make_unique<core::Buffer>(sizeInfo.accelerationStructureSize,
                                     vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                         vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                     VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  vk::AccelerationStructureCreateInfoKHR createInfo(
      {}, asBuffer->getVulkanBuffer(), {}, sizeInfo.accelerationStructureSize,
      vk::AccelerationStructureTypeKHR::eBottomLevel, {});
  auto blas = context->getDevice().createAccelerationStructureKHRUnique(createInfo);

  auto scratchBuffer = std::make_unique<core::Buffer>(
      sizeInfo.buildScratchSize,
      vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
          vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer,
      VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  vk::DeviceAddress scratchAddress =
      context->getDevice().getBufferAddress({scratchBuffer->getVulkanBuffer()});

  auto queryPool = context->getDevice().createQueryPoolUnique(
      vk::QueryPoolCreateInfo({}, vk::QueryType::eAccelerationStructureCompactedSizeKHR, 1));
  auto commandPool = context->createCommandPool();
  auto commandBuffer = commandPool->allocateCommandBuffer();

  buildInfo.scratchData.setDeviceAddress(scratchAddress);
  buildInfo.setDstAccelerationStructure(blas.get());

  commandBuffer->beginOneTime();
  commandBuffer->buildAccelerationStructures(buildInfo, mBuildRanges.data());
  commandBuffer->resetQueryPool(queryPool.get());

  if (mCompaction) {
    commandBuffer->writeAccelerationStructuresProperties(
        blas.get(), vk::QueryType::eAccelerationStructureCompactedSizeKHR, queryPool.get(), 0);
  }

  commandBuffer->end();
  commandBuffer->submitAndWait();

  if (mUpdate) {
    buildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eUpdate);
    auto updateSizeInfo = context->getDevice().getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, mMaxPrimitiveCounts);
    mUpdateScratchBuffer =
        std::make_unique<core::Buffer>(updateSizeInfo.updateScratchSize,
                                       vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                           vk::BufferUsageFlagBits::eShaderDeviceAddress |
                                           vk::BufferUsageFlagBits::eStorageBuffer,
                                       VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
    mUpdateScratchBufferAddress = mUpdateScratchBuffer->getAddress();
    logger::info("TLAS size {}, build scratch size {}, update scratch size {}",
                 sizeInfo.accelerationStructureSize, sizeInfo.buildScratchSize,
                 updateSizeInfo.updateScratchSize);
  }

  if (mCompaction) {
    auto compactionCommandBuffer = commandPool->allocateCommandBuffer();
    vk::DeviceSize compactSize{0};

    auto result = context->getDevice().getQueryPoolResults(
        queryPool.get(), 0, 1, sizeof(vk::DeviceSize), &compactSize, sizeof(vk::DeviceSize),
        vk::QueryResultFlagBits::eWait);

    if (result != vk::Result::eSuccess) {
      throw std::runtime_error("failed to get query pool result");
    }

    logger::info("BLAS original size {}, compact size {}", size, compactSize);

    if (compactSize > size) {
      throw std::runtime_error("something is wrong in compact size query!");
    }

    compactionCommandBuffer->beginOneTime();

    auto asBufferCompact =
        std::make_unique<core::Buffer>(compactSize,
                                       vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                           vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                       VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
    auto blasCompact = context->getDevice().createAccelerationStructureKHRUnique(
        vk::AccelerationStructureCreateInfoKHR(
            {}, asBufferCompact->getVulkanBuffer(), {}, compactSize,
            vk::AccelerationStructureTypeKHR::eBottomLevel, {}));

    compactionCommandBuffer->compactAccelerationStructure(blas.get(), blasCompact.get());
    compactionCommandBuffer->end();
    compactionCommandBuffer->submitAndWait();

    mBuffer = std::move(asBufferCompact);
    mAS = std::move(blasCompact);
  } else {
    mBuffer = std::move(asBuffer);
    mAS = std::move(blas);
  }
}

void BLAS::recordUpdate(
    vk::CommandBuffer commandBuffer,
    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> const &buildRanges) {
  if (!mUpdate) {
    logger::error("BLAS is not built to allow update");
  }
  vk::MemoryBarrier barrier(vk::AccessFlagBits::eTransferWrite,
                            vk::AccessFlagBits::eAccelerationStructureWriteKHR);
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {},
                                barrier, nullptr, nullptr);
  vk::AccelerationStructureBuildGeometryInfoKHR buildInfo(
      vk::AccelerationStructureTypeKHR::eBottomLevel,
      vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
      vk::BuildAccelerationStructureModeKHR::eUpdate, mAS.get(), mAS.get(), mGeometries, nullptr,
      mUpdateScratchBufferAddress);
  commandBuffer.buildAccelerationStructuresKHR(buildInfo, buildRanges.data());
}

vk::DeviceAddress BLAS::getAddress() {
  auto context = core::Context::Get();
  return context->getDevice().getAccelerationStructureAddressKHR({mAS.get()});
}

TLAS::TLAS(std::vector<vk::AccelerationStructureInstanceKHR> const &instances)
    : mInstances(instances) {}

void TLAS::build() {
  auto context = Context::Get();
  uint32_t instanceCount = static_cast<uint32_t>(mInstances.size());

  mInstanceBuffer = std::make_unique<Buffer>(
      sizeof(vk::AccelerationStructureInstanceKHR) * instanceCount,
      vk::BufferUsageFlagBits::eShaderDeviceAddress |
          vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
      VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU);
  mInstanceBuffer->upload(mInstances);

  auto commandPool = context->createCommandPool();
  auto commandBuffer = commandPool->allocateCommandBuffer();
  commandBuffer->beginOneTime();

  vk::MemoryBarrier barrier(vk::AccessFlagBits::eTransferWrite,
                            vk::AccessFlagBits::eAccelerationStructureWriteKHR);
  commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                 vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {},
                                 barrier, nullptr, nullptr);

  mInstanceBufferAddress = mInstanceBuffer->getAddress();
  vk::AccelerationStructureGeometryInstancesDataKHR instancesData(VK_FALSE,
                                                                  mInstanceBufferAddress);
  vk::AccelerationStructureGeometryKHR tlasGeometry(vk::GeometryTypeKHR::eInstances, instancesData,
                                                    {});
  vk::AccelerationStructureBuildGeometryInfoKHR buildInfo(
      vk::AccelerationStructureTypeKHR::eTopLevel,
      vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
          vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
      vk::BuildAccelerationStructureModeKHR::eBuild, nullptr, {}, tlasGeometry, {}, {});

  auto buildSizeInfo = context->getDevice().getAccelerationStructureBuildSizesKHR(
      vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, instanceCount);
  auto buildScratchBuffer = std::make_unique<core::Buffer>(
      buildSizeInfo.buildScratchSize,
      vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
          vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer,
      VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);

  buildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eUpdate);
  auto updateSizeInfo = context->getDevice().getAccelerationStructureBuildSizesKHR(
      vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, instanceCount);
  mUpdateScratchBuffer = std::make_unique<core::Buffer>(
      updateSizeInfo.updateScratchSize,
      vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
          vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer,
      VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  mUpdateScratchBufferAddress = mUpdateScratchBuffer->getAddress();

  logger::info("TLAS size {}, build scratch size {}, update scratch size {}",
               buildSizeInfo.accelerationStructureSize, buildSizeInfo.buildScratchSize,
               updateSizeInfo.updateScratchSize);

  mBuffer =
      std::make_unique<core::Buffer>(buildSizeInfo.accelerationStructureSize,
                                     vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                         vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                     VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  vk::AccelerationStructureCreateInfoKHR createInfo(
      {}, mBuffer->getVulkanBuffer(), {}, buildSizeInfo.accelerationStructureSize,
      vk::AccelerationStructureTypeKHR::eTopLevel, {});
  mAS = context->getDevice().createAccelerationStructureKHRUnique(createInfo);

  buildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
  buildInfo.setSrcAccelerationStructure(nullptr);
  buildInfo.setDstAccelerationStructure(mAS.get());
  buildInfo.scratchData.setDeviceAddress(buildScratchBuffer->getAddress());

  vk::AccelerationStructureBuildRangeInfoKHR buildRange(instanceCount, 0, 0, 0);
  commandBuffer->buildAccelerationStructures(buildInfo, &buildRange);
  commandBuffer->end();
  commandBuffer->submitAndWait();
}

void TLAS::recordUpdate(vk::CommandBuffer commandBuffer,
                        std::vector<vk::TransformMatrixKHR> const &transforms) {
  if (mInstances.size() != transforms.size()) {
    throw std::runtime_error("failed to update TLAS: length of transforms does "
                             "not match instance count");
  }

  for (size_t i = 0; i < transforms.size(); ++i) {
    mInstances[i].setTransform(transforms[i]);
  }
  mInstanceBuffer->upload(mInstances);

  mInstanceBufferAddress = mInstanceBuffer->getAddress();
  vk::AccelerationStructureGeometryInstancesDataKHR instancesData(VK_FALSE,
                                                                  mInstanceBufferAddress);
  vk::AccelerationStructureGeometryKHR tlasGeometry(vk::GeometryTypeKHR::eInstances, instancesData,
                                                    {});

  vk::AccelerationStructureBuildGeometryInfoKHR buildInfo(
      vk::AccelerationStructureTypeKHR::eTopLevel,
      vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
          vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
      vk::BuildAccelerationStructureModeKHR::eUpdate, mAS.get(), mAS.get(), tlasGeometry, {},
      mUpdateScratchBufferAddress);

  // wait for upload and BLAS builds
  vk::MemoryBarrier barrier(vk::AccessFlagBits::eTransferWrite |
                                vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                            vk::AccessFlagBits::eAccelerationStructureWriteKHR |
                                vk::AccessFlagBits::eAccelerationStructureReadKHR);
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer |
                                    vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                                vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {},
                                barrier, nullptr, nullptr);

  vk::AccelerationStructureBuildRangeInfoKHR buildRange(static_cast<uint32_t>(mInstances.size()),
                                                        0, 0, 0);
  commandBuffer.buildAccelerationStructuresKHR(buildInfo, &buildRange);
}

vk::DeviceAddress TLAS::getAddress() {
  auto context = core::Context::Get();
  return context->getDevice().getAccelerationStructureAddressKHR({mAS.get()});
}

} // namespace core
} // namespace svulkan2
