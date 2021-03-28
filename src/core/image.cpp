#include "svulkan2/core/image.h"
#include "svulkan2/core/allocator.h"
#include "svulkan2/core/buffer.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace core {

static vk::ImageType findImageTypeFromExtent(vk::Extent3D extent) {
  if (extent.depth > 1) {
    return vk::ImageType::e3D;
  }
  return vk::ImageType::e2D;
}

static vk::ImageAspectFlags findAspectBitsFromFormat(vk::Format format) {
  if (format == vk::Format::eR8Unorm) {
    return vk::ImageAspectFlagBits::eColor;
  }
  if (format == vk::Format::eR8G8B8A8Unorm) {
    return vk::ImageAspectFlagBits::eColor;
  }
  if (format == vk::Format::eR32G32B32A32Uint) {
    return vk::ImageAspectFlagBits::eColor;
  }
  if (format == vk::Format::eR32G32B32A32Sfloat) {
    return vk::ImageAspectFlagBits::eColor;
  }
  if (format == vk::Format::eR32Sfloat) {
    return vk::ImageAspectFlagBits::eColor;
  }
  if (format == vk::Format::eD32Sfloat) {
    return vk::ImageAspectFlagBits::eDepth;
  }
  if (format == vk::Format::eD24UnormS8Uint) {
    return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
  }
  throw std::runtime_error("unknown image format");
}

Image::Image(Context &context, vk::Extent3D extent, vk::Format format,
             vk::ImageUsageFlags usageFlags, VmaMemoryUsage memoryUsage,
             vk::SampleCountFlagBits sampleCount, uint32_t mipLevels,
             uint32_t arrayLayers, vk::ImageTiling tiling,
             vk::ImageCreateFlags flags)
    : mContext(&context), mExtent(extent), mFormat(format),
      mUsageFlags(usageFlags), mSampleCount(sampleCount), mMipLevels(mipLevels),
      mArrayLayers(arrayLayers), mTiling(tiling) {
  vk::ImageCreateInfo imageInfo(flags, findImageTypeFromExtent(mExtent), format,
                                extent, mipLevels, arrayLayers, sampleCount,
                                tiling, usageFlags);
  VmaAllocationCreateInfo memoryInfo{};
  memoryInfo.usage = memoryUsage;

  VmaAllocationInfo allocInfo;

  if (vmaCreateImage(context.getAllocator().getVmaAllocator(),
                     reinterpret_cast<VkImageCreateInfo *>(&imageInfo),
                     &memoryInfo, reinterpret_cast<VkImage *>(&mImage),
                     &mAllocation, &allocInfo) != VK_SUCCESS) {
    throw std::runtime_error("cannot create image");
  }

  VkMemoryPropertyFlags memFlags;
  vmaGetMemoryTypeProperties(context.getAllocator().getVmaAllocator(),
                             allocInfo.memoryType, &memFlags);
  mHostVisible = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
  mHostCoherent = (memFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
}

Image::~Image() {
  vmaDestroyImage(mContext->getAllocator().getVmaAllocator(), mImage,
                  mAllocation);
}

// void *Image::map() {
//   if (!mMapped) {
//     auto result = vmaMapMemory(mContext->getAllocator().getVmaAllocator(),
//                                mAllocation, &mMappedData);
//     if (result != VK_SUCCESS) {
//       log::critical("unable to map memory");
//       abort();
//     }
//   }
//   return mMappedData;
// }

// void Image::unmap() {
//   if (mMapped) {
//     vmaUnmapMemory(mContext->getAllocator().getVmaAllocator(), mAllocation);
//     mMapped = false;
//   }
// }

void Image::upload(void const *data, size_t size, uint32_t arrayLayer) {
  size_t imageSize = mExtent.width * mExtent.height * mExtent.depth *
                     findSizeFromFormat(mFormat);
  if (size != imageSize) {
    throw std::runtime_error("image upload failed: expecting size " +
                             std::to_string(imageSize) + ", got " +
                             std::to_string(size));
  }
  auto stagingBuffer = mContext->getAllocator().allocateStagingBuffer(size);
  stagingBuffer->upload(data, size);

  vk::BufferImageCopy copyRegion(
      0, mExtent.width, mExtent.height,
      vk::ImageSubresourceLayers(findAspectBitsFromFormat(mFormat), 0,
                                 arrayLayer, 1),
      vk::Offset3D(0, 0, 0), mExtent);

  auto cb = mContext->createCommandBuffer();
  cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  transitionLayout(cb.get(), vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eTransferDstOptimal, {},
                   vk::AccessFlagBits::eTransferWrite,
                   vk::PipelineStageFlagBits::eTopOfPipe,
                   vk::PipelineStageFlagBits::eTransfer, arrayLayer);
  cb->copyBufferToImage(stagingBuffer->getVulkanBuffer(), mImage,
                        vk::ImageLayout::eTransferDstOptimal, copyRegion);
  generateMipmaps(cb.get());
  cb->end();
  mContext->submitCommandBufferAndWait(cb.get());
}

void Image::generateMipmaps(vk::CommandBuffer cb, uint32_t arrayLayer) {
  vk::ImageMemoryBarrier barrier;
  barrier.setImage(mImage);
  barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
  barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
  barrier.subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
  barrier.subresourceRange.setBaseArrayLayer(arrayLayer);
  barrier.subresourceRange.setLayerCount(1);
  barrier.subresourceRange.setLevelCount(1);

  int32_t mipWidth = mExtent.width;
  int32_t mipHeight = mExtent.height;
  uint32_t i = 1;
  for (; i < mMipLevels; ++i) {
    // current level to next level
    barrier.subresourceRange.setBaseMipLevel(i - 1);
    barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
    barrier.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
    barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    barrier.setDstAccessMask(vk::AccessFlagBits::eTransferRead);
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                       vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
                       barrier);
    vk::ImageBlit blit(
        {vk::ImageAspectFlagBits::eColor, i - 1, 0, 1},
        {vk::Offset3D{0, 0, 0}, vk::Offset3D{mipWidth, mipHeight, 1}},
        {vk::ImageAspectFlagBits::eColor, i, 0, 1},
        {vk::Offset3D{0, 0, 0},
         vk::Offset3D{mipWidth > 1 ? mipWidth / 2 : 1,
                      mipHeight > 1 ? mipHeight / 2 : 1, 1}});
    cb.blitImage(mImage, vk::ImageLayout::eTransferSrcOptimal, mImage,
                 vk::ImageLayout::eTransferDstOptimal, blit,
                 vk::Filter::eLinear);

    // transition current level to shader read
    barrier.setOldLayout(vk::ImageLayout::eTransferSrcOptimal);
    barrier.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferRead);
    barrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                       vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {},
                       barrier);
    if (mipWidth > 1) {
      mipWidth /= 2;
    }
    if (mipHeight > 1) {
      mipHeight /= 2;
    }
  }

  // transition last level to shader read
  barrier.subresourceRange.setBaseMipLevel(i - 1);
  barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
  barrier.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
  barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
  barrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
  cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                     vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {},
                     barrier);
}

void Image::copyToBuffer(vk::Buffer buffer, size_t size, vk::Offset3D offset,
                         vk::Extent3D extent, uint32_t arrayLayer) {
  size_t imageSize =
      extent.width * extent.height * extent.depth * findSizeFromFormat(mFormat);
  if (size != imageSize) {
    throw std::runtime_error("copy to buffer failed: expecting size " +
                             std::to_string(imageSize) + ", got " +
                             std::to_string(size));
  }

  vk::ImageLayout sourceLayout;
  vk::AccessFlags sourceAccessFlag;
  vk::PipelineStageFlags sourceStage;

  vk::ImageAspectFlags aspect;
  switch (mFormat) {
  case vk::Format::eR8G8B8A8Unorm:
  case vk::Format::eR32G32B32A32Uint:
  case vk::Format::eR32G32B32A32Sfloat:
    aspect = vk::ImageAspectFlagBits::eColor;
    break;
  case vk::Format::eD32Sfloat:
    aspect = vk::ImageAspectFlagBits::eDepth;
    break;
  case vk::Format::eD24UnormS8Uint:
    vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    break;
  default:
    throw std::runtime_error("failed to download image: unsupported format.");
  }

  switch (mCurrentLayout) {
  case vk::ImageLayout::eColorAttachmentOptimal:
    sourceLayout = vk::ImageLayout::eColorAttachmentOptimal;
    sourceAccessFlag = vk::AccessFlagBits::eColorAttachmentWrite;
    sourceStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    break;
  case vk::ImageLayout::eDepthStencilAttachmentOptimal:
    sourceLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    sourceAccessFlag = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    sourceStage = vk::PipelineStageFlagBits::eEarlyFragmentTests |
                  vk::PipelineStageFlagBits::eLateFragmentTests;
    break;
  case vk::ImageLayout::eShaderReadOnlyOptimal:
    sourceLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    sourceAccessFlag = {};
    sourceStage = vk::PipelineStageFlagBits::eFragmentShader;
    break;
  case vk::ImageLayout::eTransferSrcOptimal:
    break;
  default:
    throw std::runtime_error("failed to download image: invalid layout.");
  }

  auto cb = mContext->createCommandBuffer();
  cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  if (mCurrentLayout != vk::ImageLayout::eTransferSrcOptimal) {
    transitionLayout(cb.get(), sourceLayout,
                     vk::ImageLayout::eTransferSrcOptimal, sourceAccessFlag,
                     vk::AccessFlagBits::eTransferRead, sourceStage,
                     vk::PipelineStageFlagBits::eTransfer);
  }
  vk::BufferImageCopy copyRegion(0, mExtent.width, mExtent.height,
                                 {aspect, 0, 0, 1}, offset, extent);
  cb->copyImageToBuffer(mImage, vk::ImageLayout::eTransferSrcOptimal, buffer,
                        copyRegion);
  cb->end();

  auto fence = mContext->getDevice().createFenceUnique({});
  vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eTransfer;
  mContext->getQueue().submit(
      vk::SubmitInfo(0, nullptr, &waitStage, 1, &cb.get()), fence.get());
  auto result =
      mContext->getDevice().waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
  if (result != vk::Result::eSuccess) {
    throw std::runtime_error("failed to wait for fence");
  }
  setCurrentLayout(vk::ImageLayout::eTransferSrcOptimal);
}

void Image::download(void *data, size_t size, vk::Offset3D offset,
                     vk::Extent3D extent, uint32_t arrayLayer) {
  size_t imageSize =
      extent.width * extent.height * extent.depth * findSizeFromFormat(mFormat);
  if (size != imageSize) {
    throw std::runtime_error("image download failed: expecting size " +
                             std::to_string(imageSize) + ", got " +
                             std::to_string(size));
  }

  vk::ImageLayout sourceLayout;
  vk::AccessFlags sourceAccessFlag;
  vk::PipelineStageFlags sourceStage;

  vk::ImageAspectFlags aspect;
  switch (mFormat) {
  case vk::Format::eR8G8B8A8Unorm:
  case vk::Format::eR32G32B32A32Uint:
  case vk::Format::eR32G32B32A32Sfloat:
    aspect = vk::ImageAspectFlagBits::eColor;
    break;
  case vk::Format::eD32Sfloat:
    aspect = vk::ImageAspectFlagBits::eDepth;
    break;
  case vk::Format::eD24UnormS8Uint:
    vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    break;
  default:
    throw std::runtime_error("failed to download image: unsupported format.");
  }

  switch (mCurrentLayout) {
  case vk::ImageLayout::eColorAttachmentOptimal:
    sourceLayout = vk::ImageLayout::eColorAttachmentOptimal;
    sourceAccessFlag = vk::AccessFlagBits::eColorAttachmentWrite;
    sourceStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    break;
  case vk::ImageLayout::eDepthStencilAttachmentOptimal:
    sourceLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    sourceAccessFlag = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    sourceStage = vk::PipelineStageFlagBits::eEarlyFragmentTests |
                  vk::PipelineStageFlagBits::eLateFragmentTests;
    break;
  case vk::ImageLayout::eShaderReadOnlyOptimal:
    sourceLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    sourceAccessFlag = {};
    sourceStage = vk::PipelineStageFlagBits::eFragmentShader;
    break;
  case vk::ImageLayout::eTransferSrcOptimal:
    break;
  default:
    throw std::runtime_error("failed to download image: invalid layout.");
  }

  // if (!mHostCoherent) {
  auto stagingBuffer = mContext->getAllocator().allocateStagingBuffer(size);
  auto cb = mContext->createCommandBuffer();
  cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  if (mCurrentLayout != vk::ImageLayout::eTransferSrcOptimal) {
    transitionLayout(cb.get(), sourceLayout,
                     vk::ImageLayout::eTransferSrcOptimal, sourceAccessFlag,
                     vk::AccessFlagBits::eTransferRead, sourceStage,
                     vk::PipelineStageFlagBits::eTransfer);
  }
  vk::BufferImageCopy copyRegion(0, mExtent.width, mExtent.height,
                                 {aspect, 0, 0, 1}, offset, extent);
  cb->copyImageToBuffer(mImage, vk::ImageLayout::eTransferSrcOptimal,
                        stagingBuffer->getVulkanBuffer(), copyRegion);
  cb->end();

  auto fence = mContext->getDevice().createFenceUnique({});
  vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eTransfer;
  mContext->getQueue().submit(
      vk::SubmitInfo(0, nullptr, &waitStage, 1, &cb.get()), fence.get());
  auto result =
      mContext->getDevice().waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
  if (result != vk::Result::eSuccess) {
    throw std::runtime_error("failed to wait for fence");
  }

  setCurrentLayout(vk::ImageLayout::eTransferSrcOptimal);

  std::memcpy(data, stagingBuffer->map(), size);
  stagingBuffer->unmap();
  // }
  // else {
  //   vk::ImageSubresource subResource(aspect, 0, 0);
  //   vk::SubresourceLayout subresourceLayout =
  //       mContext->getDevice().getImageSubresourceLayout(mImage, subResource);
  //   std::memcpy(data,
  //               static_cast<char const *>(map()) + subresourceLayout.offset,
  //               size);
  //   unmap();
  // }
}

void Image::download(void *data, size_t size, uint32_t arrayLayer) {
  download(data, size, {0, 0, 0}, mExtent, arrayLayer);
}

void Image::downloadPixel(void *data, size_t pixelSize, vk::Offset3D offset,
                          uint32_t arrayLayer) {
  download(data, pixelSize, offset, {1, 1, 1}, arrayLayer);
}

void Image::transitionLayout(
    vk::CommandBuffer commandBuffer, vk::ImageLayout oldImageLayout,
    vk::ImageLayout newImageLayout, vk::AccessFlags sourceAccessMask,
    vk::AccessFlags destAccessMask, vk::PipelineStageFlags sourceStage,
    vk::PipelineStageFlags destStage, uint32_t arrayLayer) {
  vk::ImageSubresourceRange imageSubresourceRange(
      findAspectBitsFromFormat(mFormat), 0, mMipLevels, arrayLayer, 1);
  vk::ImageMemoryBarrier barrier(
      sourceAccessMask, destAccessMask, oldImageLayout, newImageLayout,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, mImage,
      imageSubresourceRange);
  commandBuffer.pipelineBarrier(sourceStage, destStage, {}, nullptr, nullptr,
                                barrier);
}

void Image::transitionLayout(vk::CommandBuffer commandBuffer,
                             vk::ImageLayout oldImageLayout,
                             vk::ImageLayout newImageLayout,
                             vk::AccessFlags sourceAccessMask,
                             vk::AccessFlags destAccessMask,
                             vk::PipelineStageFlags sourceStage,
                             vk::PipelineStageFlags destStage) {
  vk::ImageSubresourceRange imageSubresourceRange(
      findAspectBitsFromFormat(mFormat), 0, mMipLevels, 0, mArrayLayers);
  vk::ImageMemoryBarrier barrier(
      sourceAccessMask, destAccessMask, oldImageLayout, newImageLayout,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, mImage,
      imageSubresourceRange);
  commandBuffer.pipelineBarrier(sourceStage, destStage, {}, nullptr, nullptr,
                                barrier);
}

} // namespace core
} // namespace svulkan2
