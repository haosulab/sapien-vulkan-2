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
  if (format == vk::Format::eR32G32B32A32Sfloat) {
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

void *Image::map() {
  if (!mMapped) {
    auto result = vmaMapMemory(mContext->getAllocator().getVmaAllocator(),
                               mAllocation, &mMappedData);
    if (result != VK_SUCCESS) {
      log::critical("unable to map memory");
      abort();
    }
  }
  return mMappedData;
}

void Image::unmap() {
  if (mMapped) {
    vmaUnmapMemory(mContext->getAllocator().getVmaAllocator(), mAllocation);
    mMapped = false;
  }
}

void Image::upload(void const *data, size_t size) {
  // TODO: handle the host visible case
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
      vk::ImageSubresourceLayers(findAspectBitsFromFormat(mFormat), 0, 0, 1),
      vk::Offset3D(0, 0, 0), mExtent);

  auto cb = mContext->createCommandBuffer();
  cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  transitionLayout(cb.get(), vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eTransferDstOptimal, {},
                   vk::AccessFlagBits::eTransferWrite,
                   vk::PipelineStageFlagBits::eTopOfPipe,
                   vk::PipelineStageFlagBits::eTransfer);
  cb->copyBufferToImage(stagingBuffer->getVulkanBuffer(), mImage,
                        vk::ImageLayout::eTransferDstOptimal, copyRegion);
  transitionLayout(cb.get(), vk::ImageLayout::eTransferDstOptimal,
                   vk::ImageLayout::eShaderReadOnlyOptimal,
                   vk::AccessFlagBits::eTransferWrite,
                   vk::AccessFlagBits::eShaderRead,
                   vk::PipelineStageFlagBits::eTransfer,
                   vk::PipelineStageFlagBits::eFragmentShader);
  cb->end();
  mContext->submitCommandBufferAndWait(cb.get());
}

void Image::download(void *data, size_t size, vk::Offset3D offset,
                     vk::Extent3D extent) {
  size_t imageSize =
      extent.width * extent.height * extent.depth * findSizeFromFormat(mFormat);
  if (size != imageSize) {
    throw std::runtime_error("image download failed: expecting size " +
                             std::to_string(imageSize) + ", got " +
                             std::to_string(size));
  }

  vk::ImageLayout sourceLayout;
  vk::AccessFlags sourceAccessFlag1;
  vk::AccessFlags sourceAccessFlag2;
  vk::PipelineStageFlags sourceStage;
  vk::ImageAspectFlags aspect;
  if (mFormat == vk::Format::eR8G8B8A8Unorm ||
      mFormat == vk::Format::eR32G32B32A32Uint ||
      mFormat == vk::Format::eR32G32B32A32Sfloat) {
    sourceLayout = vk::ImageLayout::eColorAttachmentOptimal;
    sourceAccessFlag1 = vk::AccessFlagBits::eColorAttachmentWrite;
    sourceAccessFlag2 = vk::AccessFlagBits::eColorAttachmentWrite |
                        vk::AccessFlagBits::eColorAttachmentRead;
    sourceStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    aspect = vk::ImageAspectFlagBits::eColor;
  } else if (mFormat == vk::Format::eD32Sfloat) {
    sourceLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    sourceAccessFlag1 = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    sourceAccessFlag2 = vk::AccessFlagBits::eDepthStencilAttachmentWrite |
                        vk::AccessFlagBits::eDepthStencilAttachmentRead;
    sourceStage = vk::PipelineStageFlagBits::eEarlyFragmentTests |
                  vk::PipelineStageFlagBits::eLateFragmentTests;
    aspect = vk::ImageAspectFlagBits::eDepth;
  } else if (mFormat == vk::Format::eD24UnormS8Uint) {
    sourceLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    sourceAccessFlag1 = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    sourceAccessFlag2 = vk::AccessFlagBits::eDepthStencilAttachmentWrite |
                        vk::AccessFlagBits::eDepthStencilAttachmentRead;
    sourceStage = vk::PipelineStageFlagBits::eEarlyFragmentTests |
                  vk::PipelineStageFlagBits::eLateFragmentTests;
    aspect =
        vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
  }
  if (!mHostCoherent) {
    auto stagingBuffer = mContext->getAllocator().allocateStagingBuffer(size);
    auto cb = mContext->createCommandBuffer();
    cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    transitionLayout(cb.get(), sourceLayout,
                     vk::ImageLayout::eTransferSrcOptimal, sourceAccessFlag1,
                     vk::AccessFlagBits::eTransferRead, sourceStage,
                     vk::PipelineStageFlagBits::eTransfer);
    vk::BufferImageCopy copyRegion(0, mExtent.width, mExtent.height,
                                   {aspect, 0, 0, 1}, offset, extent);
    cb->copyImageToBuffer(mImage, vk::ImageLayout::eTransferSrcOptimal,
                          stagingBuffer->getVulkanBuffer(), copyRegion);
    transitionLayout(cb.get(), vk::ImageLayout::eTransferSrcOptimal,
                     sourceLayout, vk::AccessFlagBits::eTransferRead,
                     sourceAccessFlag2, vk::PipelineStageFlagBits::eTransfer,
                     sourceStage);
    cb->end();
    mContext->submitCommandBufferAndWait(cb.get());

    std::memcpy(data, stagingBuffer->map(), size);
    stagingBuffer->unmap();
  } else {
    vk::ImageSubresource subResource(aspect, 0, 0);
    vk::SubresourceLayout subresourceLayout =
        mContext->getDevice().getImageSubresourceLayout(mImage, subResource);
    std::memcpy(data,
                static_cast<char const *>(map()) + subresourceLayout.offset,
                size);
    unmap();
  }
}

void Image::download(void *data, size_t size) {
  download(data, size, {0, 0, 0}, mExtent);
}

void Image::downloadPixel(void *data, size_t pixelSize, vk::Offset3D offset) {
  download(data, pixelSize, offset, {1, 1, 1});
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
