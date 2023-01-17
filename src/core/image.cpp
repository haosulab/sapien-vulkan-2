#include "svulkan2/core/image.h"
#include "easy/profiler.h"
#include "svulkan2/common/image.h"
#include "svulkan2/core/allocator.h"
#include "svulkan2/core/buffer.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace core {

#ifdef TRACK_ALLOCATION
static uint64_t gImageId = 1;
static uint64_t gImageCount = 0;
#endif

static vk::ImageType findImageTypeFromExtent(vk::Extent3D extent) {
  if (extent.depth > 1) {
    return vk::ImageType::e3D;
  }
  return vk::ImageType::e2D;
}

Image::Image(vk::Extent3D extent, vk::Format format,
             vk::ImageUsageFlags usageFlags, VmaMemoryUsage memoryUsage,
             vk::SampleCountFlagBits sampleCount, uint32_t mipLevels,
             uint32_t arrayLayers, vk::ImageTiling tiling,
             vk::ImageCreateFlags flags)
    : mExtent(extent), mFormat(format), mUsageFlags(usageFlags),
      mSampleCount(sampleCount), mMipLevels(mipLevels),
      mArrayLayers(arrayLayers), mTiling(tiling) {
  vk::ImageCreateInfo imageInfo(flags, findImageTypeFromExtent(mExtent), format,
                                extent, mipLevels, arrayLayers, sampleCount,
                                tiling, usageFlags);
  VmaAllocationCreateInfo memoryInfo{};
  memoryInfo.usage = memoryUsage;

  mContext = Context::Get();

  if (vmaCreateImage(mContext->getAllocator().getVmaAllocator(),
                     reinterpret_cast<VkImageCreateInfo *>(&imageInfo),
                     &memoryInfo, reinterpret_cast<VkImage *>(&mImage),
                     &mAllocation, &mAllocationInfo) != VK_SUCCESS) {
    throw std::runtime_error("cannot create image");
  }

  VkMemoryPropertyFlags memFlags;
  vmaGetMemoryTypeProperties(mContext->getAllocator().getVmaAllocator(),
                             mAllocationInfo.memoryType, &memFlags);
  mHostVisible = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
  mHostCoherent = (memFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

#ifdef TRACK_ALLOCATION
  mImageId = gImageId++;
  log::info("Create Image {}; Total {}", mImageId, ++gImageCount);
#endif
}

Image::~Image() {
  vmaDestroyImage(mContext->getAllocator().getVmaAllocator(), mImage,
                  mAllocation);
#ifdef TRACK_ALLOCATION
  log::info("Destroy Image {}, Total {}", mImageId, --gImageCount);
#endif
}

void Image::uploadLevel(void const *data, size_t size, uint32_t arrayLayer,
                        uint32_t mipLevel) {
  auto extent = computeMipLevelExtent(mExtent, mipLevel);
  size_t imageSize =
      extent.width * extent.height * extent.depth * getFormatSize(mFormat);
  if (size != imageSize) {
    throw std::runtime_error("image upload failed: expecting size " +
                             std::to_string(imageSize) + ", got " +
                             std::to_string(size));
  }
  auto stagingBuffer = mContext->getAllocator().allocateStagingBuffer(size);
  stagingBuffer->upload(data, size);

  vk::BufferImageCopy copyRegion(
      0, extent.width, extent.height,
      vk::ImageSubresourceLayers(getImageAspectFlags(mFormat), mipLevel,
                                 arrayLayer, 1),
      vk::Offset3D(0, 0, 0), extent);

  auto pool = mContext->createCommandPool();
  auto cb = pool->allocateCommandBuffer();
  cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  cb->copyBufferToImage(stagingBuffer->getVulkanBuffer(), mImage,
                        vk::ImageLayout::eTransferDstOptimal, copyRegion);
  cb->end();
  mContext->getQueue().submitAndWait(cb.get());
}

void Image::upload(void const *data, size_t size, uint32_t arrayLayer,
                   bool mipmaps) {
  size_t imageSize =
      mExtent.width * mExtent.height * mExtent.depth * getFormatSize(mFormat);
  if (size != imageSize) {
    throw std::runtime_error("image upload failed: expecting size " +
                             std::to_string(imageSize) + ", got " +
                             std::to_string(size));
  }
  auto stagingBuffer = mContext->getAllocator().allocateStagingBuffer(size);
  stagingBuffer->upload(data, size);

  vk::BufferImageCopy copyRegion(
      0, mExtent.width, mExtent.height,
      vk::ImageSubresourceLayers(getImageAspectFlags(mFormat), 0, arrayLayer,
                                 1),
      vk::Offset3D(0, 0, 0), mExtent);

  auto pool = mContext->createCommandPool();
  auto cb = pool->allocateCommandBuffer();
  cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  transitionLayout(cb.get(), vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eTransferDstOptimal, {},
                   vk::AccessFlagBits::eTransferWrite,
                   vk::PipelineStageFlagBits::eTopOfPipe,
                   vk::PipelineStageFlagBits::eTransfer, arrayLayer);
  cb->copyBufferToImage(stagingBuffer->getVulkanBuffer(), mImage,
                        vk::ImageLayout::eTransferDstOptimal, copyRegion);
  if (mipmaps) {
    generateMipmaps(cb.get(), arrayLayer);
  } else {
    vk::ImageMemoryBarrier barrier;
    barrier.setImage(mImage);
    barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
    barrier.subresourceRange.setBaseArrayLayer(arrayLayer);
    barrier.subresourceRange.setLayerCount(1);
    barrier.subresourceRange.setBaseMipLevel(0);
    barrier.subresourceRange.setLevelCount(1);
    barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
    barrier.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    barrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    cb->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {},
                        barrier);
  }
  cb->end();
  mContext->getQueue().submitAndWait(cb.get());
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

void Image::recordCopyToBuffer(vk::CommandBuffer cb, vk::Buffer buffer,
                               size_t bufferOffset, size_t size,
                               vk::Offset3D offset, vk::Extent3D extent,
                               uint32_t arrayLayer) {
  size_t imageSize =
      extent.width * extent.height * extent.depth * getFormatSize(mFormat);

  if (size != imageSize) {
    throw std::runtime_error("copy to buffer failed: expecting size " +
                             std::to_string(imageSize) + ", got " +
                             std::to_string(size));
  }

  if (mCurrentLayout == vk::ImageLayout::eGeneral) {
    // wait for everything in general layout
    transitionLayout(cb, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                     vk::AccessFlagBits::eMemoryWrite,
                     vk::AccessFlagBits::eTransferRead,
                     vk::PipelineStageFlagBits::eAllCommands,
                     vk::PipelineStageFlagBits::eTransfer);
  } else if (mCurrentLayout != vk::ImageLayout::eTransferSrcOptimal) {
    // guess what to wait and transition to TransferSrcOptimal
    vk::ImageLayout sourceLayout = vk::ImageLayout::eUndefined;
    vk::AccessFlags sourceAccessFlag{};
    vk::PipelineStageFlags sourceStage{};

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
      sourceAccessFlag = vk::AccessFlagBits::eShaderRead;
      sourceStage = vk::PipelineStageFlagBits::eFragmentShader;
      break;
    case vk::ImageLayout::eTransferSrcOptimal:
      break;
    default:
      throw std::runtime_error("failed to download image: invalid layout.");
    }
    transitionLayout(cb, sourceLayout, vk::ImageLayout::eTransferSrcOptimal,
                     sourceAccessFlag, vk::AccessFlagBits::eTransferRead,
                     sourceStage, vk::PipelineStageFlagBits::eTransfer);
  }

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

  vk::BufferImageCopy copyRegion(bufferOffset, mExtent.width, mExtent.height,
                                 {aspect, 0, 0, 1}, offset, extent);
  cb.copyImageToBuffer(mImage, mCurrentLayout, buffer, copyRegion);
}

void Image::recordCopyFromBuffer(vk::CommandBuffer cb, vk::Buffer buffer,
                                 size_t bufferOffset, size_t size,
                                 vk::Offset3D offset, vk::Extent3D extent,
                                 uint32_t arrayLayer) {
  size_t imageSize =
      extent.width * extent.height * extent.depth * getFormatSize(mFormat);

  if (size != imageSize) {
    throw std::runtime_error("buffer copy to image failed: expecting size " +
                             std::to_string(imageSize) + ", got " +
                             std::to_string(size));
  }

  if (mCurrentLayout != vk::ImageLayout::eGeneral &&
      mCurrentLayout != vk::ImageLayout::eTransferDstOptimal) {

    vk::ImageLayout sourceLayout = vk::ImageLayout::eUndefined;
    vk::AccessFlags sourceAccessFlag{};
    vk::PipelineStageFlags sourceStage{};

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
      sourceAccessFlag = vk::AccessFlagBits::eShaderRead;
      sourceStage = vk::PipelineStageFlagBits::eFragmentShader;
      break;
    case vk::ImageLayout::eTransferSrcOptimal:
      sourceLayout = vk::ImageLayout::eTransferSrcOptimal;
      sourceAccessFlag = vk::AccessFlagBits::eTransferWrite;
      sourceStage = vk::PipelineStageFlagBits::eTransfer;
      break;
    case vk::ImageLayout::eTransferDstOptimal:
      break;
    default:
      throw std::runtime_error("failed to download image: invalid layout.");
    }
    transitionLayout(cb, sourceLayout, vk::ImageLayout::eTransferDstOptimal,
                     sourceAccessFlag, vk::AccessFlagBits::eTransferWrite,
                     sourceStage, vk::PipelineStageFlagBits::eTransfer);
  }

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

  vk::BufferImageCopy copyRegion(bufferOffset, mExtent.width, mExtent.height,
                                 {aspect, 0, 0, 1}, offset, extent);
  cb.copyBufferToImage(buffer, mImage, mCurrentLayout, copyRegion);
}

void Image::copyToBuffer(vk::Buffer buffer, size_t size, vk::Offset3D offset,
                         vk::Extent3D extent, uint32_t arrayLayer) {
  auto pool = mContext->createCommandPool();
  auto cb = pool->allocateCommandBuffer();
  cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  recordCopyToBuffer(cb.get(), buffer, 0, size, offset, extent, arrayLayer);
  cb->end();
  vk::Result result = mContext->getQueue().submitAndWait(cb.get());
  if (result != vk::Result::eSuccess) {
    throw std::runtime_error("failed to wait for fence");
  }
}

void Image::download(void *data, size_t size, vk::Offset3D offset,
                     vk::Extent3D extent, uint32_t arrayLayer,
                     uint32_t mipLevel) {
  EASY_FUNCTION();

  size_t imageSize =
      extent.width * extent.height * extent.depth * getFormatSize(mFormat);

  if (size != imageSize) {
    throw std::runtime_error("image download failed: expecting size " +
                             std::to_string(imageSize) + ", got " +
                             std::to_string(size));
  }

  vk::ImageLayout sourceLayout = vk::ImageLayout::eUndefined;
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
    aspect =
        vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    break;
  default:
    throw std::runtime_error("failed to download image: unsupported format.");
  }

  EASY_BLOCK("Record command buffer");
  auto pool = mContext->createCommandPool();
  auto cb = pool->allocateCommandBuffer();
  cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  if (mCurrentLayout == vk::ImageLayout::eGeneral) {
    // wait for everything in general layout
    transitionLayout(
        cb.get(), vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
        vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eTransferRead,
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eTransfer);
  } else if (mCurrentLayout != vk::ImageLayout::eTransferSrcOptimal) {
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

    transitionLayout(cb.get(), sourceLayout,
                     vk::ImageLayout::eTransferSrcOptimal, sourceAccessFlag,
                     vk::AccessFlagBits::eTransferRead, sourceStage,
                     vk::PipelineStageFlagBits::eTransfer);
  }

  EASY_BLOCK("Allocating staging buffer");
  auto stagingBuffer =
      mContext->getAllocator().allocateStagingBuffer(size, true);
  EASY_END_BLOCK;
  vk::BufferImageCopy copyRegion(0, extent.width, extent.height,
                                 {aspect, mipLevel, arrayLayer, 1}, offset,
                                 extent);
  cb->copyImageToBuffer(mImage, mCurrentLayout,
                        stagingBuffer->getVulkanBuffer(), copyRegion);
  cb->end();
  EASY_END_BLOCK;

  EASY_BLOCK("Submit and wait");
  vk::Result result = mContext->getQueue().submitAndWait(cb.get());

  if (result != vk::Result::eSuccess) {
    throw std::runtime_error("failed to wait for fence");
  }
  EASY_END_BLOCK;

  setCurrentLayout(vk::ImageLayout::eTransferSrcOptimal);

  EASY_BLOCK("Copy data to CPU");
  std::memcpy(data, stagingBuffer->map(), size);
  stagingBuffer->unmap();
  EASY_END_BLOCK;
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
  vk::ImageSubresourceRange imageSubresourceRange(getImageAspectFlags(mFormat),
                                                  0, mMipLevels, arrayLayer, 1);
  vk::ImageMemoryBarrier barrier(
      sourceAccessMask, destAccessMask, oldImageLayout, newImageLayout,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, mImage,
      imageSubresourceRange);
  commandBuffer.pipelineBarrier(sourceStage, destStage, {}, nullptr, nullptr,
                                barrier);
  mCurrentLayout = newImageLayout;
}

void Image::transitionLayout(vk::CommandBuffer commandBuffer,
                             vk::ImageLayout oldImageLayout,
                             vk::ImageLayout newImageLayout,
                             vk::AccessFlags sourceAccessMask,
                             vk::AccessFlags destAccessMask,
                             vk::PipelineStageFlags sourceStage,
                             vk::PipelineStageFlags destStage) {
  vk::ImageSubresourceRange imageSubresourceRange(
      getImageAspectFlags(mFormat), 0, mMipLevels, 0, mArrayLayers);
  vk::ImageMemoryBarrier barrier(
      sourceAccessMask, destAccessMask, oldImageLayout, newImageLayout,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, mImage,
      imageSubresourceRange);
  commandBuffer.pipelineBarrier(sourceStage, destStage, {}, nullptr, nullptr,
                                barrier);
  mCurrentLayout = newImageLayout;
}

#ifdef SVULKAN2_CUDA_INTEROP
vk::DeviceSize Image::getRowPitch() {
  vk::ImageSubresource subresource(vk::ImageAspectFlagBits::eColor, 0, 0);
  vk::SubresourceLayout layout =
      mContext->getDevice().getImageSubresourceLayout(mImage, subresource);
  return layout.rowPitch;
}

cudaMipmappedArray_t Image::getCudaArray() {
  if (mCudaArray) {
    return mCudaArray;
  }
  mCudaDeviceId =
      getCudaDeviceIdFromPhysicalDevice(mContext->getPhysicalDevice());
  if (mCudaDeviceId < 0) {
    throw std::runtime_error(
        "Vulkan Device is not visible to CUDA. You probably need to unset the "
        "CUDA_VISIBLE_DEVICES variable. Or you can try other "
        "CUDA_VISIBLE_DEVICES until you find a working one.");
  }

  checkCudaErrors(cudaSetDevice(mCudaDeviceId));
  cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
  externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  externalMemoryHandleDesc.size =
      mAllocationInfo.offset + mAllocationInfo.size; // TODO check

  vk::MemoryGetFdInfoKHR vkMemoryGetFdInfoKHR;
  vkMemoryGetFdInfoKHR.setPNext(nullptr);
  vkMemoryGetFdInfoKHR.setMemory(mAllocationInfo.deviceMemory);
  vkMemoryGetFdInfoKHR.setHandleType(
      vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);

  auto cudaFd = mContext->getDevice().getMemoryFdKHR(vkMemoryGetFdInfoKHR);
  externalMemoryHandleDesc.handle.fd = cudaFd;

  checkCudaErrors(
      cudaImportExternalMemory(&mCudaMem, &externalMemoryHandleDesc));

  cudaExternalMemoryMipmappedArrayDesc desc = {};
  desc.extent.width = mExtent.width;
  desc.extent.height = mExtent.height;
  desc.extent.depth = mExtent.depth;
  desc.numLevels = mMipLevels;
  desc.offset = mAllocationInfo.offset;
  desc.flags = 0;
  if (mFormat == vk::Format::eR32G32B32A32Sfloat) {
    desc.formatDesc.x = 32;
    desc.formatDesc.y = 32;
    desc.formatDesc.z = 32;
    desc.formatDesc.w = 32;
    desc.formatDesc.f = cudaChannelFormatKindFloat;
  } else if (mFormat == vk::Format::eR32G32B32Sfloat) {
    desc.formatDesc.x = 32;
    desc.formatDesc.y = 32;
    desc.formatDesc.z = 32;
    desc.formatDesc.w = 0;
    desc.formatDesc.f = cudaChannelFormatKindFloat;
  }

  // TODO: free!!!
  checkCudaErrors(
      cudaExternalMemoryGetMappedMipmappedArray(&mCudaArray, mCudaMem, &desc));
  return mCudaArray;
}

int Image::getCudaDeviceId() {
  if (!mCudaArray) {
    return getCudaDeviceIdFromPhysicalDevice(mContext->getPhysicalDevice());
  }
  return mCudaDeviceId;
}

#endif

} // namespace core
} // namespace svulkan2
