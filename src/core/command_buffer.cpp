#include "svulkan2/core/command_buffer.h"
#include "svulkan2/core/command_pool.h"
#include "svulkan2/core/device.h"
#include "svulkan2/core/queue.h"

namespace svulkan2 {
namespace core {

CommandBuffer::CommandBuffer(std::shared_ptr<CommandPool> pool) : mCommandPool(pool) {
  mCommandBuffer =
      std::move(mCommandPool->getDevice()
                    ->getInternal()
                    .allocateCommandBuffersUnique(
                        {mCommandPool->getInternal(), vk::CommandBufferLevel::ePrimary, 1})
                    .front());
}

void CommandBuffer::reset() { mCommandBuffer->reset(); }

void CommandBuffer::begin() { mCommandBuffer->begin(vk::CommandBufferBeginInfo({}, {})); }
void CommandBuffer::beginOneTime() {
  mCommandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
}

void CommandBuffer::end() { mCommandBuffer->end(); }

void CommandBuffer::beginRenderPass(vk::RenderPass pass, vk::Framebuffer framebuffer,
                                    vk::Rect2D renderArea,
                                    vk::ArrayProxy<vk::ClearValue> clearValues) {
  std::vector<vk::ClearValue> values = {clearValues.begin(), clearValues.end()};
  vk::RenderPassBeginInfo info{pass, framebuffer, renderArea, values};
  mCommandBuffer->beginRenderPass(info, vk::SubpassContents::eInline);
}

void CommandBuffer::endRenderPass() { mCommandBuffer->endRenderPass(); }

void CommandBuffer::bindPipeline(vk::PipelineBindPoint point, vk::Pipeline pipeline) {
  mCommandBuffer->bindPipeline(point, pipeline);
}

void CommandBuffer::bindDescriptorSets(vk::PipelineBindPoint point, vk::PipelineLayout layout,
                                       uint32_t index,
                                       const vk::ArrayProxy<const vk::DescriptorSet> &sets) {
  mCommandBuffer->bindDescriptorSets(point, layout, index, sets, {});
}

void CommandBuffer::bindDescriptorSet(vk::PipelineBindPoint point, vk::PipelineLayout layout,
                                      uint32_t index, vk::DescriptorSet set) {
  mCommandBuffer->bindDescriptorSets(point, layout, index, set, {});
}

void CommandBuffer::bindVertexBuffer(vk::Buffer buffer) {
  mCommandBuffer->bindVertexBuffers(0, buffer, {0});
}

void CommandBuffer::bindIndexBuffer(vk::Buffer buffer) {
  mCommandBuffer->bindIndexBuffer(buffer, 0, vk::IndexType::eUint32);
}

void CommandBuffer::draw(uint32_t count) { mCommandBuffer->draw(count, 1, 0, 0); }

void CommandBuffer::drawIndexed(uint32_t count) { mCommandBuffer->drawIndexed(count, 1, 0, 0, 0); }

void CommandBuffer::setViewport(float x, float y, float width, float height, float minz,
                                float maxz) {
  vk::Viewport v(x, y, width, height, minz, maxz);
  mCommandBuffer->setViewport(0, v);
}

void CommandBuffer::setScissor(int32_t x, int32_t y, uint32_t width, uint32_t height) {
  vk::Rect2D scissor{{x, y}, {width, height}};
  mCommandBuffer->setScissor(0, scissor);
}

void CommandBuffer::pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stageFlags,
                                  uint32_t offset, uint32_t size, const void *pValues) {
  mCommandBuffer->pushConstants(layout, stageFlags, offset, size, pValues);
}

void CommandBuffer::blitImage(vk::Image srcImage, vk::ImageLayout srcImageLayout,
                              vk::Image dstImage, vk::ImageLayout dstImageLayout,
                              const vk::ArrayProxy<const vk::ImageBlit> &regions,
                              vk::Filter filter) {
  mCommandBuffer->blitImage(srcImage, srcImageLayout, dstImage, dstImageLayout, regions, filter);
}

void CommandBuffer::copyBufferToImage(vk::Buffer srcBuffer, vk::Image dstImage,
                                      vk::ImageLayout dstImageLayout,
                                      const vk::ArrayProxy<const vk::BufferImageCopy> &regions) {
  mCommandBuffer->copyBufferToImage(srcBuffer, dstImage, dstImageLayout, regions);
}

void CommandBuffer::copyImageToBuffer(vk::Image srcImage, vk::ImageLayout srcImageLayout,
                                      vk::Buffer dstBuffer,
                                      const vk::ArrayProxy<const vk::BufferImageCopy> &regions) {
  mCommandBuffer->copyImageToBuffer(srcImage, srcImageLayout, dstBuffer, regions);
}

void CommandBuffer::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                               const vk::ArrayProxy<const vk::BufferCopy> &regions) {
  mCommandBuffer->copyBuffer(srcBuffer, dstBuffer, regions);
}

void CommandBuffer::pipelineBarrier(
    vk::PipelineStageFlags srcStageMask, vk::PipelineStageFlags dstStageMask,
    vk::DependencyFlags dependencyFlags,
    const vk::ArrayProxy<const vk::MemoryBarrier> &memoryBarriers,
    const vk::ArrayProxy<const vk::BufferMemoryBarrier> &bufferMemoryBarriers,
    const vk::ArrayProxy<const vk::ImageMemoryBarrier> &imageMemoryBarriers) {
  mCommandBuffer->pipelineBarrier(srcStageMask, dstStageMask, dependencyFlags, memoryBarriers,
                                  bufferMemoryBarriers, imageMemoryBarriers);
}

void CommandBuffer::resetQueryPool(vk::QueryPool queryPool, uint32_t firstQuery,
                                   uint32_t queryCount) {
  mCommandBuffer->resetQueryPool(queryPool, firstQuery, queryCount);
}

void CommandBuffer::buildAccelerationStructures(
    const vk::ArrayProxy<const vk::AccelerationStructureBuildGeometryInfoKHR> &infos,
    const vk::ArrayProxy<const vk::AccelerationStructureBuildRangeInfoKHR *const>
        &pBuildRangeInfos) {
  mCommandBuffer->buildAccelerationStructuresKHR(infos, pBuildRangeInfos);
}

void CommandBuffer::writeAccelerationStructuresProperties(
    const vk::ArrayProxy<const vk::AccelerationStructureKHR> &accelerationStructures,
    vk::QueryType queryType, vk::QueryPool queryPool, uint32_t firstQuery) {
  mCommandBuffer->writeAccelerationStructuresPropertiesKHR(accelerationStructures, queryType,
                                                           queryPool, firstQuery);
}

void CommandBuffer::compactAccelerationStructure(vk::AccelerationStructureKHR src,
                                                 vk::AccelerationStructureKHR dst) {
  mCommandBuffer->copyAccelerationStructureKHR(
      {src, dst, vk::CopyAccelerationStructureModeKHR::eCompact});
}

void CommandBuffer::traceRays(const vk::StridedDeviceAddressRegionKHR &raygenShaderBindingTable,
                              const vk::StridedDeviceAddressRegionKHR &missShaderBindingTable,
                              const vk::StridedDeviceAddressRegionKHR &hitShaderBindingTable,
                              const vk::StridedDeviceAddressRegionKHR &callableShaderBindingTable,
                              uint32_t width, uint32_t height, uint32_t depth) {
  mCommandBuffer->traceRaysKHR(raygenShaderBindingTable, missShaderBindingTable,
                               hitShaderBindingTable, callableShaderBindingTable, width, height,
                               depth);
}

void CommandBuffer::dispatch(uint32_t x, uint32_t y, uint32_t z) {
  mCommandBuffer->dispatch(x, y, z);
}

void CommandBuffer::submit() {
  mCommandPool->getDevice()->getQueue().submit(mCommandBuffer.get(), vk::Fence{});
}

vk::Result CommandBuffer::submitAndWait() {
  return mCommandPool->getDevice()->getQueue().submitAndWait(mCommandBuffer.get());
}

CommandBuffer::~CommandBuffer() {}

} // namespace core
} // namespace svulkan2
