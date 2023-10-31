/**
 * @file image.h
 * @brief GPU resources for a Vulkan image
 * @author Fanbo Xiang
 * Contact: fxiang@eng.ucsd.edu
 */

#pragma once
#include "svulkan2/common/vk.h"
#include <memory>

typedef struct cudaMipmappedArray *cudaMipmappedArray_t;
typedef struct CUexternalMemory_st *cudaExternalMemory_t;

class ktxVulkanTexture;

namespace svulkan2 {
namespace core {

class Image {
private:
  std::shared_ptr<class Context> mContext;
  vk::ImageType mType;
  vk::Extent3D mExtent;
  vk::Format mFormat;
  vk::ImageUsageFlags mUsageFlags;
  vk::SampleCountFlagBits mSampleCount;
  uint32_t mMipLevels;
  uint32_t mArrayLayers;
  vk::ImageTiling mTiling;

  vk::Image mImage;
  VmaAllocation mAllocation;
  VmaAllocationInfo mAllocationInfo;

  std::unique_ptr<ktxVulkanTexture> mKtxTexture;

  std::vector<vk::ImageLayout> mCurrentLayerLayout;

  void generateMipmaps(vk::CommandBuffer cb, uint32_t arrayLayer = 0);

public:
  Image(vk::ImageType type, vk::Extent3D extent, vk::Format format, vk::ImageUsageFlags usageFlags,
        VmaMemoryUsage memoryUsage,
        vk::SampleCountFlagBits sampleCount = vk::SampleCountFlagBits::e1, uint32_t mipLevels = 1,
        uint32_t arrayLayers = 1, vk::ImageTiling tiling = vk::ImageTiling::eOptimal,
        vk::ImageCreateFlags flags = {});
  Image(std::unique_ptr<ktxVulkanTexture> ktxImage);

  Image(const Image &) = delete;
  Image(const Image &&) = delete;
  Image &operator=(const Image &) = delete;
  Image &operator=(Image &&) = delete;

  ~Image();

  vk::Image getVulkanImage() const { return mImage; }

  void uploadLevel(void const *data, size_t size, uint32_t arrayLayer, uint32_t mipLevel);

  void upload(void const *data, size_t size, uint32_t arrayLayer = 0, bool mipmaps = true);
  template <typename DataType>
  void upload(std::vector<DataType> const &data, uint32_t arrayLayer = 0, bool mipmaps = true) {
    upload(data.data(), data.size() * sizeof(DataType), arrayLayer, mipmaps);
  }

  void copyToBuffer(vk::Buffer buffer, size_t size, vk::Offset3D offset, vk::Extent3D extent,
                    uint32_t arrayLayer = 0);
  void recordCopyToBuffer(vk::CommandBuffer cb, vk::Buffer buffer, size_t bufferOffset,
                          size_t size, vk::Offset3D offset, vk::Extent3D extent,
                          uint32_t arrayLayer = 0);
  void recordCopyFromBuffer(vk::CommandBuffer cb, vk::Buffer buffer, size_t bufferOffset,
                            size_t size, vk::Offset3D offset, vk::Extent3D extent,
                            uint32_t arrayLayer = 0);

  void download(void *data, size_t size, vk::Offset3D offset, vk::Extent3D extent,
                uint32_t arrayLayer = 0, uint32_t mipLevel = 0);
  void download(void *data, size_t size, uint32_t arrayLayer = 0);
  void downloadPixel(void *data, size_t pixelSize, vk::Offset3D offset, uint32_t arrayLayer = 0);

  template <typename DataType>
  std::vector<DataType> download(vk::Offset3D offset, vk::Extent3D extent,
                                 uint32_t arrayLayer = 0) {
    if (!isFormatCompatible<DataType>(mFormat)) {
      throw std::runtime_error("Download failed: incompatible format");
    }
    static_assert(sizeof(DataType) == 1 ||
                  sizeof(DataType) == 4); // only support char, int or float
    size_t size = getFormatSize(mFormat) * extent.width * extent.height * extent.depth;
    std::vector<DataType> data(size / sizeof(DataType));
    download(data.data(), size, offset, extent, arrayLayer);
    return data;
  }
  template <typename DataType> std::vector<DataType> download(uint32_t arrayLayer = 0) {
    return download<DataType>({0, 0, 0}, mExtent, arrayLayer);
  }
  template <typename DataType>
  std::vector<DataType> downloadPixel(vk::Offset3D offset, uint32_t arrayLayer = 0) {
    if (!isFormatCompatible<DataType>(mFormat)) {
      throw std::runtime_error("Download pixel failed: incompatible format");
    }
    return download<DataType>(offset, {1, 1, 1}, arrayLayer);
  }

  void setCurrentLayout(vk::ImageLayout layout);
  void setCurrentLayout(uint32_t layer, vk::ImageLayout layout);
  vk::ImageLayout getCurrentLayout(uint32_t layer) const;

  void transitionLayout(vk::CommandBuffer commandBuffer, vk::ImageLayout oldImageLayout,
                        vk::ImageLayout newImageLayout, vk::AccessFlags sourceAccessMask,
                        vk::AccessFlags destAccessMask, vk::PipelineStageFlags sourceStage,
                        vk::PipelineStageFlags destStage, uint32_t arrayLayer);

  void transitionLayout(vk::CommandBuffer commandBuffer, vk::ImageLayout oldImageLayout,
                        vk::ImageLayout newImageLayout, vk::AccessFlags sourceAccessMask,
                        vk::AccessFlags destAccessMask, vk::PipelineStageFlags sourceStage,
                        vk::PipelineStageFlags destStage);

  inline vk::ImageType getType() const { return mType; }
  inline vk::Extent3D getExtent() const { return mExtent; }
  inline vk::Format const &getFormat() const { return mFormat; }
  inline uint32_t getMipLevels() const { return mMipLevels; }
  inline uint32_t getArrayLayers() const { return mArrayLayers; }
  inline vk::ImageUsageFlags getUsage() const { return mUsageFlags; }
  inline vk::SampleCountFlagBits getSampleCount() const { return mSampleCount; }
  inline vk::ImageTiling getTiling() const { return mTiling; }

#ifdef SVULKAN2_CUDA_INTEROP
private:
  // TODO: cuda feature is not finished, do not use
  cudaMipmappedArray_t mCudaArray{};
  cudaExternalMemory_t mCudaMem{};
  int mCudaDeviceId{-1};

public:
  cudaMipmappedArray_t getCudaArray();
  int getCudaDeviceId();
  vk::DeviceSize getRowPitch();
#endif
};

} // namespace core
} // namespace svulkan2
