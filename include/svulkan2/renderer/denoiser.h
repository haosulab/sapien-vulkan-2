#ifdef SVULKAN2_CUDA_INTEROP

#pragma once
#include "svulkan2/common/log.h"
#include "svulkan2/core/buffer.h"
#include "svulkan2/core/command_pool.h"
#include "svulkan2/core/image.h"
#include <cstdio>
#include <cuda.h>
#include <memory>
#include <optix.h>

static_assert(OPTIX_VERSION == 70600);

namespace svulkan2 {
namespace renderer {

class DenoiserOptix {
public:
  bool init(OptixPixelFormat pixelFormat, bool albedo, bool normal, bool hdr);
  void allocate(uint32_t width, uint32_t height);
  void free();
  void denoise(core::Image &color, core::Image *albedo, core::Image *normal);

  core::Buffer &getDenoisedBuffer();

  inline uint32_t getWidth() const { return mWidth; }
  inline uint32_t getHeight() const { return mHeight; }

  ~DenoiserOptix();

private:
  class Context {
  public:
    Context();
    ~Context();

    OptixDeviceContext optixDevice{};
  };

  static std::weak_ptr<Context> gContext;
  std::shared_ptr<Context> mContext;

  CUstream mCudaStream{};
  OptixDenoiser mDenoiser{};
  OptixDenoiserOptions mOptions{};

  bool useAlbedo();
  bool useNormal();

  OptixPixelFormat mPixelFormat{};
  uint32_t mPixelSize{};
  OptixDenoiserSizes mSizes{};

  OptixDenoiserParams mParams{};

  CUdeviceptr mStatePtr{};
  CUdeviceptr mScratchPtr{};

  OptixDenoiserGuideLayer mGuideLayer;
  OptixDenoiserLayer mLayer;

  uint32_t mWidth{};
  uint32_t mHeight{};

  std::unique_ptr<core::Buffer> mInputBuffer;
  CUdeviceptr mInputPtr{};
  std::unique_ptr<core::Buffer> mOutputBuffer;
  CUdeviceptr mOutputPtr{};

  std::unique_ptr<core::Buffer> mAlbedoBuffer;
  CUdeviceptr mAlbedoPtr{};
  std::unique_ptr<core::Buffer> mNormalBuffer;
  CUdeviceptr mNormalPtr{};

  std::unique_ptr<core::CommandPool> mCommandPool;
  vk::UniqueCommandBuffer mCommandBufferIn;
  vk::UniqueCommandBuffer mCommandBufferOut;

  vk::UniqueSemaphore mSem{};
  cudaExternalSemaphore_t mCudaSem{};
  uint64_t mSemValue{0};
};
} // namespace renderer
}; // namespace svulkan2

#endif
