#ifdef CUDA_INTEROP
#include "svulkan2/core/cuda_buffer.h"
#include "svulkan2/core/context.h"

#define checkCudaErrors(call)                                                  \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace svulkan2 {
namespace core {

uint32_t findMemoryType(vk::PhysicalDevice physicalDevice,
                        uint32_t typeFilter) {
  auto memProps = physicalDevice.getMemoryProperties();
  for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
    if (typeFilter & (1 << i)) {
      return i;
    }
  }
  throw std::runtime_error("failed to find memory type");
}

CudaBuffer::CudaBuffer(Context &context, vk::DeviceSize size,
                       vk::BufferUsageFlags usageFlags)
    : mContext(&context), mSize(size) {
  mMemory = mContext->getDevice().allocateMemoryUnique(vk::MemoryAllocateInfo(
      size, findMemoryType(mContext->getPhysicalDevice(),
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)));
  mBuffer = mContext->getDevice().createBufferUnique(
      vk::BufferCreateInfo({}, size, usageFlags));

  context.getDevice().bindBufferMemory(mBuffer.get(), mMemory.get(), 0);

  cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
  externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  externalMemoryHandleDesc.size = size;

  int fd = -1;
  vk::MemoryGetFdInfoKHR vkMemoryGetFdInfoKHR;
  vkMemoryGetFdInfoKHR.setPNext(nullptr);
  vkMemoryGetFdInfoKHR.setMemory(mMemory.get());
  vkMemoryGetFdInfoKHR.setHandleType(
      vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);

  PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
  fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)mContext->getDevice().getProcAddr(
      "vkGetMemoryFdKHR");
  if (!fpGetMemoryFdKHR) {
    throw std::runtime_error("failed to retrieve vkGetMemoryFdKHR");
  }
  if (fpGetMemoryFdKHR(
          mContext->getDevice(),
          reinterpret_cast<VkMemoryGetFdInfoKHR *>(&vkMemoryGetFdInfoKHR),
          &fd) != VK_SUCCESS) {
    throw std::runtime_error("failed to retrieve handle for buffer");
  }
  externalMemoryHandleDesc.handle.fd = fd;
  checkCudaErrors(
      cudaImportExternalMemory(&mCudaMem, &externalMemoryHandleDesc));

  cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
  externalMemBufferDesc.offset = 0;
  externalMemBufferDesc.size = size;
  externalMemBufferDesc.flags = 0;

  checkCudaErrors(cudaExternalMemoryGetMappedBuffer(&mCudaPtr, mCudaMem,
                                                    &externalMemBufferDesc));
}

CudaBuffer::~CudaBuffer() {
  checkCudaErrors(cudaDestroyExternalMemory(mCudaMem));
  checkCudaErrors(cudaFree(mCudaPtr));
}

} // namespace core
} // namespace svulkan2

#endif
