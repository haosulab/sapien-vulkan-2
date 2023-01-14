#ifdef SVULKAN2_CUDA_INTEROP

#pragma once
#include "svulkan2/core/buffer.h"
#include "svulkan2/core/command_pool.h"
#include "svulkan2/core/image.h"
#include <cstdio>
#include <cuda.h>
#include <memory>
#include <optix.h>

#define checkOptix(call)                                                       \
  do {                                                                         \
    auto val = (call);                                                         \
    if (val != OPTIX_SUCCESS) {                                                \
      fprintf(stderr, "%s:%d: Optix call failed: %d", __FILE__, __LINE__,      \
              val);                                                            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (false)

#define checkCudaRuntime(call)                                                 \
  do {                                                                         \
    auto val = (call);                                                         \
    if (val != cudaSuccess) {                                                  \
      fprintf(stderr, "%s:%d: CUDA runtime call failed", __FILE__, __LINE__);  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (false)

#define checkCudaDriver(call)                                                  \
  do {                                                                         \
    auto val = (call);                                                         \
    if (val != CUDA_SUCCESS) {                                                 \
      fprintf(stderr, "%s:%d: CUDA driver call failed %d", __FILE__, __LINE__, \
              val);                                                            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (false)

namespace svulkan2 {
namespace renderer {

class DenoiserOptix {
public:
  bool init(OptixPixelFormat pixelFormat, bool albedo, bool normal, bool hdr);
  void allocate(uint32_t width, uint32_t height);
  void denoise(core::Image &color, core::Image *albedo, core::Image *normal);

  core::Buffer &getDenoisedBuffer();

private:
  class Context {
  public:
    Context();
    ~Context();

    OptixDeviceContext optixDevice{};
    void *libcuda;

    CUresult (*cuStreamCreate)(CUstream *, unsigned int);
  };

  static std::weak_ptr<Context> gContext;
  std::shared_ptr<Context> mContext;

  CUstream mCudaStream{};
  OptixDenoiser mDenoiser{};
  OptixDenoiserOptions mOptions{};
  OptixPixelFormat mPixelFormat{};
  uint32_t mPixelSize{};
  OptixDenoiserAlphaMode mAlphaMode{};
  OptixDenoiserSizes mSizes{};

  OptixDenoiserParams mParams;

  CUdeviceptr mStatePtr{};
  CUdeviceptr mScratchPtr{};
  CUdeviceptr mMinRgbPtr{};
  CUdeviceptr mIntensityPtr{};

  OptixDenoiserGuideLayer mGuideLayer;
  OptixDenoiserLayer mLayer;

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
