#ifdef SVULKAN2_CUDA_INTEROP
#include "svulkan2/renderer/denoiser.h"
#include "optix_function_table_definition.h"
#include "svulkan2/common/log.h"
#include "svulkan2/core/context.h"
#include <cuda_runtime.h>
#include <filesystem>
#include <optional>
#include <optix_stubs.h>
#include <vector>

namespace svulkan2 {
namespace renderer {

DenoiserOptix::Context::Context() {
  libcuda = dlopen("libcuda.so", RTLD_LOCAL | RTLD_NOW);
  if (!libcuda) {
    throw std::runtime_error("failed to load cuda driver");
  }
  if (cudaFree(0) != cudaSuccess) {
    throw std::runtime_error("failed to init cuda runtime");
  }
  if (optixInit() != OPTIX_SUCCESS) {
    throw std::runtime_error("failed to init optix");
  }
  checkOptix(optixDeviceContextCreate(nullptr, nullptr, &optixDevice));

  this->cuStreamCreate = reinterpret_cast<decltype(this->cuStreamCreate)>(
      dlsym(libcuda, "cuStreamCreate"));
  log::info("optix initialized");
}

DenoiserOptix::Context::~Context() {
  optixDeviceContextDestroy(optixDevice);
  if (libcuda) {
    dlclose(libcuda);
  }
}

std::weak_ptr<DenoiserOptix::Context> DenoiserOptix::gContext = {};

bool DenoiserOptix::init(OptixPixelFormat pixelFormat, bool albedo, bool normal,
                         bool hdr) {

  if (!(mContext = gContext.lock())) {
    gContext = mContext = std::make_shared<Context>();
  }

  mCommandPool = core::Context::Get()->createCommandPool();
  mCommandBufferIn = mCommandPool->allocateCommandBuffer();
  mCommandBufferOut = mCommandPool->allocateCommandBuffer();

  checkCudaDriver(mContext->cuStreamCreate(&mCudaStream, CU_STREAM_DEFAULT));

  mOptions.guideAlbedo = albedo;
  mOptions.guideNormal = normal;

  optixDenoiserCreate(mContext->optixDevice,
                      hdr ? OPTIX_DENOISER_MODEL_KIND_HDR
                          : OPTIX_DENOISER_MODEL_KIND_LDR,
                      &mOptions, &mDenoiser);

  mPixelFormat = pixelFormat;
  switch (pixelFormat) {
  case OPTIX_PIXEL_FORMAT_FLOAT3:
    mPixelSize = static_cast<uint32_t>(3 * sizeof(float));
    mAlphaMode = OPTIX_DENOISER_ALPHA_MODE_COPY;
    break;
  case OPTIX_PIXEL_FORMAT_FLOAT4:
    mPixelSize = static_cast<uint32_t>(4 * sizeof(float));
    mAlphaMode = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
    break;
  case OPTIX_PIXEL_FORMAT_UCHAR3:
    mPixelSize = static_cast<uint32_t>(3 * sizeof(uint8_t));
    mAlphaMode = OPTIX_DENOISER_ALPHA_MODE_COPY;
    break;
  case OPTIX_PIXEL_FORMAT_UCHAR4:
    mPixelSize = static_cast<uint32_t>(4 * sizeof(uint8_t));
    mAlphaMode = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
    break;
  case OPTIX_PIXEL_FORMAT_HALF3:
    mPixelSize = static_cast<uint32_t>(3 * sizeof(uint16_t));
    mAlphaMode = OPTIX_DENOISER_ALPHA_MODE_COPY;
    break;
  case OPTIX_PIXEL_FORMAT_HALF4:
    mPixelSize = static_cast<uint32_t>(4 * sizeof(uint16_t));
    mAlphaMode = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
    break;
  default:
    throw std::runtime_error("Unsupported OPTIX_PIXEL_FORMAT!");
    break;
  }

  return true;
}

void DenoiserOptix::allocate(uint32_t width, uint32_t height) {
  checkOptix(
      optixDenoiserComputeMemoryResources(mDenoiser, width, height, &mSizes));

  checkCudaRuntime(cudaMalloc((void **)&mStatePtr, mSizes.stateSizeInBytes));
  checkCudaRuntime(cudaMalloc((void **)&mScratchPtr,
                              mSizes.withoutOverlapScratchSizeInBytes));
  checkCudaRuntime(cudaMalloc((void **)&mMinRgbPtr, 4 * sizeof(float)));

  checkOptix(optixDenoiserSetup(mDenoiser, mCudaStream, width, height,
                                mStatePtr, mSizes.stateSizeInBytes, mScratchPtr,
                                mSizes.withoutOverlapScratchSizeInBytes));
  checkCudaRuntime(cudaMalloc((void **)&mIntensityPtr, sizeof(float)));

  mInputBuffer =
      std::make_unique<core::Buffer>(width * height * mPixelSize,
                                     vk::BufferUsageFlagBits::eTransferDst |
                                         vk::BufferUsageFlagBits::eTransferSrc,
                                     VMA_MEMORY_USAGE_GPU_ONLY);
  auto ptr = mInputBuffer->getCudaPtr();
  std::memcpy(&mInputPtr, &ptr, sizeof(ptr));

  mOutputBuffer =
      std::make_unique<core::Buffer>(width * height * mPixelSize,
                                     vk::BufferUsageFlagBits::eTransferDst |
                                         vk::BufferUsageFlagBits::eTransferSrc,
                                     VMA_MEMORY_USAGE_GPU_ONLY);
  ptr = mOutputBuffer->getCudaPtr();
  std::memcpy(&mOutputPtr, &ptr, sizeof(ptr));

  if (mOptions.guideAlbedo) {
    mAlbedoBuffer = std::make_unique<core::Buffer>(
        width * height * mPixelSize,
        vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_GPU_ONLY);
    ptr = mAlbedoBuffer->getCudaPtr();
    std::memcpy(&mAlbedoPtr, &ptr, sizeof(ptr));
  }

  if (mOptions.guideNormal) {
    mNormalBuffer = std::make_unique<core::Buffer>(
        width * height * mPixelSize,
        vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_GPU_ONLY);
    ptr = mNormalBuffer->getCudaPtr();
    std::memcpy(&mNormalPtr, &ptr, sizeof(ptr));
  }

  vk::Device device = core::Context::Get()->getDevice();

  // create timeline semaphore with value 0
  vk::SemaphoreTypeCreateInfo timelineCreateInfo(vk::SemaphoreType::eTimeline,
                                                 0);
  vk::SemaphoreCreateInfo createInfo{};
  vk::ExportSemaphoreCreateInfo exportCreateInfo(
      vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd);
  createInfo.setPNext(&exportCreateInfo);
  exportCreateInfo.setPNext(&timelineCreateInfo);
  mSem = device.createSemaphoreUnique(createInfo);

  // export vulkan semaphore as cuda semaphore
  int fd = device.getSemaphoreFdKHR(
      {mSem.get(), vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd});
  cudaExternalSemaphoreHandleDesc desc = {};
  desc.flags = 0;
  desc.handle.fd = fd;
  desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  checkCudaRuntime(cudaImportExternalSemaphore(&mCudaSem, &desc));

  // cudaDestroyExternalSemaphore(mCudaSem);  TODO
}

void DenoiserOptix::denoise(core::Image &color, core::Image *albedo,
                            core::Image *normal) {
  mCommandBufferIn->reset();
  mCommandBufferIn->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  color.recordCopyToBuffer(
      mCommandBufferIn.get(), mInputBuffer->getVulkanBuffer(), 0,
      mInputBuffer->getSize(), {0, 0, 0}, color.getExtent());
  if (albedo) {
    albedo->recordCopyToBuffer(
        mCommandBufferIn.get(), mAlbedoBuffer->getVulkanBuffer(), 0,
        mAlbedoBuffer->getSize(), {0, 0, 0}, color.getExtent());
  }
  if (normal) {
    normal->recordCopyToBuffer(
        mCommandBufferIn.get(), mNormalBuffer->getVulkanBuffer(), 0,
        mNormalBuffer->getSize(), {0, 0, 0}, color.getExtent());
  }
  mCommandBufferIn->end();

  core::Context::Get()->getQueue().submit(mCommandBufferIn.get(), {}, {}, {},
                                          mSem.get(), ++mSemValue, {});

  cudaExternalSemaphoreWaitParams waitParams{};
  waitParams.params.fence.value = mSemValue;
  cudaWaitExternalSemaphoresAsync(&mCudaSem, &waitParams, 1, mCudaStream);

  uint32_t width = color.getExtent().width;
  uint32_t height = color.getExtent().height;

  mLayer.input = OptixImage2D{mInputPtr,          width,      height,
                              mPixelSize * width, mPixelSize, mPixelFormat};
  mLayer.output = OptixImage2D{mOutputPtr,         width,      height,
                               mPixelSize * width, mPixelSize, mPixelFormat};
  mLayer.previousOutput = {0};

  if (mIntensityPtr) {
    checkOptix(optixDenoiserComputeIntensity(
        mDenoiser, mCudaStream, &mLayer.input, mIntensityPtr, mScratchPtr,
        mSizes.withoutOverlapScratchSizeInBytes));
  }

  mParams.blendFactor = 0.f;
  mParams.denoiseAlpha = mAlphaMode;
  mParams.hdrIntensity = mIntensityPtr;
  mParams.hdrAverageColor = 0;
  mParams.temporalModeUsePreviousLayers = 0;

  mGuideLayer.flow = {};
  mGuideLayer.outputInternalGuideLayer = {};
  mGuideLayer.previousOutputInternalGuideLayer = {};
  mGuideLayer.albedo = {};
  mGuideLayer.normal = {};
  if (mOptions.guideAlbedo && albedo) {
    mGuideLayer.albedo =
        OptixImage2D{mAlbedoPtr,         width,      height,
                     mPixelSize * width, mPixelSize, mPixelFormat};
  }
  if (mOptions.guideNormal && normal) {
    mGuideLayer.normal =
        OptixImage2D{mNormalPtr,         width,      height,
                     mPixelSize * width, mPixelSize, mPixelFormat};
  }

  checkOptix(optixDenoiserInvoke(mDenoiser, mCudaStream, &mParams, mStatePtr,
                                 mSizes.stateSizeInBytes, &mGuideLayer, &mLayer,
                                 1, 0, 0, mScratchPtr,
                                 mSizes.withoutOverlapScratchSizeInBytes));

  cudaExternalSemaphoreSignalParams sigParams{};
  sigParams.flags = 0;
  sigParams.params.fence.value = ++mSemValue;
  cudaSignalExternalSemaphoresAsync(&mCudaSem, &sigParams, 1, mCudaStream);

  mCommandBufferOut->reset();
  mCommandBufferOut->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  color.recordCopyFromBuffer(
      mCommandBufferOut.get(), mOutputBuffer->getVulkanBuffer(), 0,
      mOutputBuffer->getSize(), {0, 0, 0}, color.getExtent());
  mCommandBufferOut->end();

  vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eTransfer;
  core::Context::Get()->getQueue().submit(mCommandBufferOut.get(), mSem.get(),
                                          waitStage, mSemValue, {}, {}, {});
}

} // namespace renderer
} // namespace svulkan2

#endif
