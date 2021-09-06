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

// int getPCIBusIdFromCudaDeviceId(int cudaDeviceId) {
//   static std::unordered_map<int, int> pciBusIdToDeviceId;

//   if (!pciBusIdToDeviceId.contains(cudaDeviceId)) {
//     int pciBusId;
//     std::string pciBus(20, '\0');
//     cudaDeviceGetPCIBusId(pciBus.data(), 20, cudaDeviceId);

//     if (pciBus[0] == '\0') // invalid cudaDeviceId
//       pciBusId = -1;
//     else {
//       std::stringstream ss;
//       ss << std::hex << pciBus.substr(5, 2);
//       ss >> pciBusId;
//     }
//     pciBusIdToDeviceId[cudaDeviceId] = pciBusId;
//   }

//   return pciBusIdToDeviceId[cudaDeviceId];
// }

// int getCudaDeviceIdFromPhysicalDevice(const vk::PhysicalDevice &device) {
//   vk::PhysicalDeviceProperties2KHR p2;
//   vk::PhysicalDevicePCIBusInfoPropertiesEXT pciInfo;
//   pciInfo.pNext = p2.pNext;
//   p2.pNext = &pciInfo;
//   device.getProperties2(&p2);

//   for (int cudaDeviceId = 0; cudaDeviceId < 20; cudaDeviceId++)
//     if (static_cast<int>(pciInfo.pciBus) ==
//         getPCIBusIdFromCudaDeviceId(cudaDeviceId))
//       return cudaDeviceId;

//   return -1; // should never reach here
// }

CudaBuffer::CudaBuffer(std::shared_ptr<Context> context, vk::DeviceSize size,
                       vk::BufferUsageFlags usageFlags)
    : mContext(context), mSize(size) {
  mMemory = mContext->getDevice().allocateMemoryUnique(vk::MemoryAllocateInfo(
      size, findMemoryType(mContext->getPhysicalDevice(),
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)));
  mBuffer = mContext->getDevice().createBufferUnique(
      vk::BufferCreateInfo({}, size, usageFlags));

  mContext->getDevice().bindBufferMemory(mBuffer.get(), mMemory.get(), 0);

  mCudaDeviceId =
      getCudaDeviceIdFromPhysicalDevice(mContext->getPhysicalDevice());

  if (mCudaDeviceId < 0) {
    throw std::runtime_error("Vulkan Device is not visible to CUDA. You probably need to unset the CUDA_VISIBLE_DEVICES variable. Or you can try other CUDA_VISIBLE_DEVICES until you find a working one.");
  }

  checkCudaErrors(cudaSetDevice(mCudaDeviceId));

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
