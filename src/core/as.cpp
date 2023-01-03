#include "svulkan2/core/as.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace core {

BLAS::BLAS(std::vector<vk::AccelerationStructureGeometryKHR> const &geometries,
           std::vector<vk::AccelerationStructureBuildRangeInfoKHR> const
               &buildRanges) {
  auto context = Context::Get();

  // vk::AccelerationStructureBuildGeometryInfoKHR buildInfo(
  //     vk::AccelerationStructureTypeKHR::eBottomLevel,
  //     vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction,
  //     vk::BuildAccelerationStructureModeKHR::eBuild, nullptr,
  //     vk::AccelerationStructureKHR{}, geometries, nullptr,
  //     vk::DeviceOrHostAddressKHR{});
  // vk::AccelerationStructureBuildSizesInfoKHR sizeInfo =
  //     context->getDevice().getAccelerationStructureBuildSizesKHR(
  //         vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo,
  //         maxPrimitiveCounts);
  // vk::DeviceSize size = sizeInfo.accelerationStructureSize;

  // mBuffer = std::make_unique<core::Buffer>(
  //     sizeInfo.accelerationStructureSize,
  //     vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
  //         vk::BufferUsageFlagBits::eShaderDeviceAddress,
  //     VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  // mAS = context->getDevice().createAccelerationStructureKHRUnique(
  //     vk::AccelerationStructureCreateInfoKHR(
  //         {}, mBuffer->getVulkanBuffer(), {}, size,
  //         vk::AccelerationStructureTypeKHR::eBottomLevel, {}));
}

void BLAS::build(bool compaction) {
  auto context = Context::Get();

  std::vector<uint32_t> maxPrimitiveCounts;
  for (auto &range : mBuildRanges) {
    maxPrimitiveCounts.push_back(range.primitiveCount);
  }

  vk::AccelerationStructureBuildGeometryInfoKHR buildInfo(
      vk::AccelerationStructureTypeKHR::eBottomLevel,
      compaction ? vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate
                 : vk::BuildAccelerationStructureFlagsKHR{},
      vk::BuildAccelerationStructureModeKHR::eBuild, nullptr,
      vk::AccelerationStructureKHR{}, mGeometries, nullptr,
      vk::DeviceOrHostAddressKHR{});
  vk::AccelerationStructureBuildSizesInfoKHR sizeInfo =
      context->getDevice().getAccelerationStructureBuildSizesKHR(
          vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo,
          maxPrimitiveCounts);
  vk::DeviceSize size = sizeInfo.accelerationStructureSize;

  auto asBuffer = std::make_unique<core::Buffer>(
      sizeInfo.accelerationStructureSize,
      vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
          vk::BufferUsageFlagBits::eShaderDeviceAddress,
      VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  vk::AccelerationStructureCreateInfoKHR createInfo(
      {}, asBuffer->getVulkanBuffer(), {}, sizeInfo.accelerationStructureSize,
      vk::AccelerationStructureTypeKHR::eBottomLevel, {});
  auto blas =
      context->getDevice().createAccelerationStructureKHRUnique(createInfo);

  auto scratchBuffer = std::make_unique<core::Buffer>(
      sizeInfo.buildScratchSize,
      vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
          vk::BufferUsageFlagBits::eShaderDeviceAddress,
      VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  vk::DeviceAddress scratchAddress =
      context->getDevice().getBufferAddress({scratchBuffer->getVulkanBuffer()});

  auto queryPool =
      context->getDevice().createQueryPoolUnique(vk::QueryPoolCreateInfo(
          {}, vk::QueryType::eAccelerationStructureCompactedSizeKHR, 1));
  auto commandPool = context->createCommandPool();
  auto commandBuffer = commandPool->allocateCommandBuffer();

  std::vector<const vk::AccelerationStructureBuildRangeInfoKHR *>
      pBuildRangeInfos;
  for (auto &range : mBuildRanges) {
    pBuildRangeInfos.push_back(&range);
  }

  buildInfo.scratchData.setDeviceAddress(scratchAddress);
  commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  commandBuffer->buildAccelerationStructuresKHR(buildInfo, pBuildRangeInfos);
  commandBuffer->resetQueryPool(queryPool.get(), 0, 1);

  if (compaction) {
    commandBuffer->writeAccelerationStructuresPropertiesKHR(
        blas.get(), vk::QueryType::eAccelerationStructureCompactedSizeKHR,
        queryPool.get(), 0);
  }

  commandBuffer->end();
  context->getQueue().submitAndWait(commandBuffer.get());

  if (compaction) {
    auto compactionCommandBuffer = commandPool->allocateCommandBuffer();
    auto result = context->getDevice().getQueryPoolResult<vk::DeviceSize>(
        queryPool.get(), 0, 1, sizeof(vk::DeviceSize),
        vk::QueryResultFlagBits::eWait);
    vk::DeviceSize compactSize = result.value;
    if (result.result != vk::Result::eSuccess) {
      throw std::runtime_error("failed to get query pool result");
    }

    compactionCommandBuffer->begin(
        {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    auto asBufferCompact = std::make_unique<core::Buffer>(
        sizeInfo.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
    auto blasCompact =
        context->getDevice().createAccelerationStructureKHRUnique(
            vk::AccelerationStructureCreateInfoKHR(
                {}, asBuffer->getVulkanBuffer(), {}, compactSize,
                vk::AccelerationStructureTypeKHR::eBottomLevel, {}));

    vk::CopyAccelerationStructureInfoKHR copyInfo(
        blas.get(), blasCompact.get(),
        vk::CopyAccelerationStructureModeKHR::eCompact);
    compactionCommandBuffer->copyAccelerationStructureKHR(copyInfo);
    compactionCommandBuffer->end();

    context->getQueue().submitAndWait(compactionCommandBuffer.get());
    log::info("BLAS compaction original size {}, compact size {}", size,
              compactSize);

    mBuffer = std::move(asBufferCompact);
    mAS = std::move(blasCompact);
  } else {
    mBuffer = std::move(asBuffer);
    mAS = std::move(blas);
  }
}

} // namespace core
} // namespace svulkan2
