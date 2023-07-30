#pragma once
#include "svulkan2/common/vk.h"
#include <memory>

namespace svulkan2 {
namespace core {
class Context;
class CommandPool;

class CommandBuffer {
public:
  CommandBuffer(std::shared_ptr<CommandPool>);

  inline std::shared_ptr<CommandPool> getCommandPool() const { return mCommandPool; }
  inline vk::CommandBuffer const &getInternal() const { return mCommandBuffer.get(); }

  void reset();

  void begin();
  void beginOneTime();
  void end();

  void beginRenderPass(vk::RenderPass pass, vk::Framebuffer framebuffer, vk::Rect2D renderArea,
                       vk::ArrayProxy<vk::ClearValue> clearValues);
  void endRenderPass();

  void bindPipeline(vk::PipelineBindPoint, vk::Pipeline);

  void bindDescriptorSets(vk::PipelineBindPoint, vk::PipelineLayout, uint32_t index,
                          const vk::ArrayProxy<const vk::DescriptorSet> &sets);

  void bindDescriptorSet(vk::PipelineBindPoint, vk::PipelineLayout, uint32_t index,
                         vk::DescriptorSet);

  void bindVertexBuffer(vk::Buffer); // TODO: handle multiple buffers
  void bindIndexBuffer(vk::Buffer);  // TODO: handle index types
  void draw(uint32_t count);         // TODO: support instanced drawing
  void drawIndexed(uint32_t count);  // TODO: support instanced drawing

  void setViewport(float x, float y, float width, float height, float minz, float maxz);
  void setScissor(int32_t x, int32_t y, uint32_t width, uint32_t height);

  void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stageFlags, uint32_t offset,
                     uint32_t size, const void *pValues);

  void blitImage(vk::Image srcImage, vk::ImageLayout srcImageLayout, vk::Image dstImage,
                 vk::ImageLayout dstImageLayout,
                 const vk::ArrayProxy<const vk::ImageBlit> &regions, vk::Filter filter);

  void copyBufferToImage(vk::Buffer srcBuffer, vk::Image dstImage, vk::ImageLayout dstImageLayout,
                         const vk::ArrayProxy<const vk::BufferImageCopy> &regions);
  void copyImageToBuffer(vk::Image srcImage, vk::ImageLayout srcImageLayout, vk::Buffer dstBuffer,
                         const vk::ArrayProxy<const vk::BufferImageCopy> &regions);
  void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                  const vk::ArrayProxy<const vk::BufferCopy> &regions);

  // barriers
  void pipelineBarrier(vk::PipelineStageFlags srcStageMask, vk::PipelineStageFlags dstStageMask,
                       vk::DependencyFlags dependencyFlags,
                       const vk::ArrayProxy<const vk::MemoryBarrier> &memoryBarriers,
                       const vk::ArrayProxy<const vk::BufferMemoryBarrier> &bufferMemoryBarriers,
                       const vk::ArrayProxy<const vk::ImageMemoryBarrier> &imageMemoryBarriers);

  void resetQueryPool(vk::QueryPool queryPool, uint32_t firstQuery = 0, uint32_t queryCount = 1);

  void buildAccelerationStructures(
      const vk::ArrayProxy<const vk::AccelerationStructureBuildGeometryInfoKHR> &infos,
      const vk::ArrayProxy<const vk::AccelerationStructureBuildRangeInfoKHR *const>
          &pBuildRangeInfos);
  void writeAccelerationStructuresProperties(
      const vk::ArrayProxy<const vk::AccelerationStructureKHR> &accelerationStructures,
      vk::QueryType queryType, vk::QueryPool queryPool, uint32_t firstQuery);
  void compactAccelerationStructure(vk::AccelerationStructureKHR src,
                                    vk::AccelerationStructureKHR dst);

  void traceRays(const vk::StridedDeviceAddressRegionKHR &raygenShaderBindingTable,
                 const vk::StridedDeviceAddressRegionKHR &missShaderBindingTable,
                 const vk::StridedDeviceAddressRegionKHR &hitShaderBindingTable,
                 const vk::StridedDeviceAddressRegionKHR &callableShaderBindingTable,
                 uint32_t width, uint32_t height, uint32_t depth = 1);

  void dispatch(uint32_t x, uint32_t y, uint32_t z);

  void submit(); // TODO: synchronization
  vk::Result submitAndWait();

  CommandBuffer(CommandBuffer const &) = delete;
  CommandBuffer &operator=(CommandBuffer const &) = delete;
  CommandBuffer(CommandBuffer const &&) = delete;
  CommandBuffer &operator=(CommandBuffer const &&) = delete;
  ~CommandBuffer();

private:
  std::shared_ptr<CommandPool> mCommandPool;
  vk::UniqueCommandBuffer mCommandBuffer;
};

} // namespace core
} // namespace svulkan2
