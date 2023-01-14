#include "svulkan2/core/as.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace core {

BLAS::BLAS(
    std::vector<vk::AccelerationStructureGeometryKHR> const &geometries,
    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> const &buildRanges)
    : mGeometries(geometries), mBuildRanges(buildRanges) {}

void BLAS::build(bool compaction) {
  auto context = Context::Get();

  std::vector<uint32_t> maxPrimitiveCounts;
  for (auto &range : mBuildRanges) {
    maxPrimitiveCounts.push_back(range.primitiveCount);
  }

  vk::AccelerationStructureBuildGeometryInfoKHR buildInfo(
      vk::AccelerationStructureTypeKHR::eBottomLevel,
      compaction ? vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction
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
          vk::BufferUsageFlagBits::eShaderDeviceAddress |
          vk::BufferUsageFlagBits::eStorageBuffer,
      VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  vk::DeviceAddress scratchAddress =
      context->getDevice().getBufferAddress({scratchBuffer->getVulkanBuffer()});

  auto queryPool =
      context->getDevice().createQueryPoolUnique(vk::QueryPoolCreateInfo(
          {}, vk::QueryType::eAccelerationStructureCompactedSizeKHR, 1));
  auto commandPool = context->createCommandPool();
  auto commandBuffer = commandPool->allocateCommandBuffer();

  // std::vector<const vk::AccelerationStructureBuildRangeInfoKHR *>
  //     pBuildRangeInfos;
  // for (auto &range : mBuildRanges) {
  //   pBuildRangeInfos.push_back(&range);
  // }

  buildInfo.scratchData.setDeviceAddress(scratchAddress);
  buildInfo.setDstAccelerationStructure(blas.get());

  commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  // TODO: break here
  commandBuffer->buildAccelerationStructuresKHR(buildInfo, mBuildRanges.data());
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
    vk::DeviceSize compactSize{0};

    // FIXME: somehow getQueryPoolResult<vk::DeviceSize> (without s) is not
    // working everything else is fine it does not make sense...

    auto result = context->getDevice().getQueryPoolResults(
        queryPool.get(), 0, 1, sizeof(vk::DeviceSize), &compactSize,
        sizeof(vk::DeviceSize), vk::QueryResultFlagBits::eWait);

    if (result != vk::Result::eSuccess) {
      throw std::runtime_error("failed to get query pool result");
    }

    log::info("BLAS original size {}, compact size {}", size, compactSize);

    if (compactSize > size) {
      throw std::runtime_error("something is wrong in compact size query!");
    }

    compactionCommandBuffer->begin(
        {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    auto asBufferCompact = std::make_unique<core::Buffer>(
        compactSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
    auto blasCompact =
        context->getDevice().createAccelerationStructureKHRUnique(
            vk::AccelerationStructureCreateInfoKHR(
                {}, asBufferCompact->getVulkanBuffer(), {}, compactSize,
                vk::AccelerationStructureTypeKHR::eBottomLevel, {}));

    vk::CopyAccelerationStructureInfoKHR copyInfo(
        blas.get(), blasCompact.get(),
        vk::CopyAccelerationStructureModeKHR::eCompact);
    compactionCommandBuffer->copyAccelerationStructureKHR(copyInfo);
    compactionCommandBuffer->end();

    context->getQueue().submitAndWait(compactionCommandBuffer.get());

    mBuffer = std::move(asBufferCompact);
    mAS = std::move(blasCompact);
  } else {
    mBuffer = std::move(asBuffer);
    mAS = std::move(blas);
  }
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

  mUpdateCommandPool = context->createCommandPool();
  auto commandBuffer = mUpdateCommandPool->allocateCommandBuffer();
  commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  vk::MemoryBarrier barrier(vk::AccessFlagBits::eTransferWrite,
                            vk::AccessFlagBits::eAccelerationStructureWriteKHR);
  commandBuffer->pipelineBarrier(
      vk::PipelineStageFlagBits::eTransfer,
      vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, barrier,
      nullptr, nullptr);

  mInstanceBufferAddress = mInstanceBuffer->getAddress();
  vk::AccelerationStructureGeometryInstancesDataKHR instancesData(
      VK_FALSE, mInstanceBufferAddress);
  vk::AccelerationStructureGeometryKHR tlasGeometry(
      vk::GeometryTypeKHR::eInstances, instancesData, {});
  vk::AccelerationStructureBuildGeometryInfoKHR buildInfo(
      vk::AccelerationStructureTypeKHR::eTopLevel,
      vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
          vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
      vk::BuildAccelerationStructureModeKHR::eBuild, nullptr, {}, tlasGeometry,
      {}, {});

  auto buildSizeInfo =
      context->getDevice().getAccelerationStructureBuildSizesKHR(
          vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo,
          instanceCount);
  auto buildScratchBuffer = std::make_unique<core::Buffer>(
      buildSizeInfo.buildScratchSize,
      vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
          vk::BufferUsageFlagBits::eShaderDeviceAddress |
          vk::BufferUsageFlagBits::eStorageBuffer,
      VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);

  buildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eUpdate);
  auto updateSizeInfo =
      context->getDevice().getAccelerationStructureBuildSizesKHR(
          vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo,
          instanceCount);
  mUpdateScratchBuffer = std::make_unique<core::Buffer>(
      updateSizeInfo.updateScratchSize,
      vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
          vk::BufferUsageFlagBits::eShaderDeviceAddress |
          vk::BufferUsageFlagBits::eStorageBuffer,
      VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  mUpdateScratchBufferAddress = mUpdateScratchBuffer->getAddress();

  log::info("TLAS size {}, build scratch size {}, update scratch size {}",
            buildSizeInfo.accelerationStructureSize,
            buildSizeInfo.buildScratchSize, updateSizeInfo.updateScratchSize);

  mBuffer = std::make_unique<core::Buffer>(
      buildSizeInfo.accelerationStructureSize,
      vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
          vk::BufferUsageFlagBits::eShaderDeviceAddress,
      VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  vk::AccelerationStructureCreateInfoKHR createInfo(
      {}, mBuffer->getVulkanBuffer(), {},
      buildSizeInfo.accelerationStructureSize,
      vk::AccelerationStructureTypeKHR::eTopLevel, {});
  mAS = context->getDevice().createAccelerationStructureKHRUnique(createInfo);

  buildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
  buildInfo.setSrcAccelerationStructure(nullptr);
  buildInfo.setDstAccelerationStructure(mAS.get());
  buildInfo.scratchData.setDeviceAddress(buildScratchBuffer->getAddress());

  vk::AccelerationStructureBuildRangeInfoKHR buildRange(instanceCount, 0, 0, 0);
  commandBuffer->buildAccelerationStructuresKHR(buildInfo, &buildRange);
  commandBuffer->end();
  context->getQueue().submitAndWait(commandBuffer.get());
}

void TLAS::update(std::vector<vk::TransformMatrixKHR> const &transforms) {
  if (mInstances.size() != transforms.size()) {
    throw std::runtime_error("failed to update TLAS: length of transforms does "
                             "not match instance count");
  }

  for (size_t i = 0; i < transforms.size(); ++i) {
    mInstances[i].setTransform(transforms[i]);
  }
  mInstanceBuffer->upload(mInstances);

  mInstanceBufferAddress = mInstanceBuffer->getAddress();
  vk::AccelerationStructureGeometryInstancesDataKHR instancesData(
      VK_FALSE, mInstanceBufferAddress);
  vk::AccelerationStructureGeometryKHR tlasGeometry(
      vk::GeometryTypeKHR::eInstances, instancesData, {});

  vk::AccelerationStructureBuildGeometryInfoKHR buildInfo(
      vk::AccelerationStructureTypeKHR::eTopLevel,
      vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
          vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
      vk::BuildAccelerationStructureModeKHR::eUpdate, mAS.get(), mAS.get(),
      tlasGeometry, {}, mUpdateScratchBufferAddress);

  auto commandBuffer = mUpdateCommandPool->allocateCommandBuffer();
  commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  // wait for upload
  vk::MemoryBarrier barrier(vk::AccessFlagBits::eTransferWrite,
                            vk::AccessFlagBits::eAccelerationStructureWriteKHR);
  commandBuffer->pipelineBarrier(
      vk::PipelineStageFlagBits::eTransfer,
      vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, barrier,
      nullptr, nullptr);

  vk::AccelerationStructureBuildRangeInfoKHR buildRange(
      static_cast<uint32_t>(mInstances.size()), 0, 0, 0);
  commandBuffer->buildAccelerationStructuresKHR(buildInfo, &buildRange);

  commandBuffer->end();

  Context::Get()->getQueue().submitAndWait(commandBuffer.get());
}

vk::DeviceAddress TLAS::getAddress() {
  auto context = core::Context::Get();
  return context->getDevice().getAccelerationStructureAddressKHR({mAS.get()});
}

} // namespace core
} // namespace svulkan2
