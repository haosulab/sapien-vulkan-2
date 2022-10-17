#ifdef SVULKAN2_CUDA_INTEROP
#pragma once

#include "vk.h"
#include <map>
#include <string>
#include <cuda_runtime.h>

inline int getPCIBusIdFromCudaDeviceId(int cudaDeviceId) {
  int pciBusId = -1;
  std::string pciBus(20, '\0');
  cudaDeviceGetPCIBusId(pciBus.data(), 20, cudaDeviceId);

  if (pciBus[0] == '\0') // invalid cudaDeviceId
    pciBusId = -1;
  else {
    std::stringstream ss;
    ss << std::hex << pciBus.substr(5, 2);
    ss >> pciBusId;
  }

  return pciBusId;
}

inline int getCudaDeviceIdFromPhysicalDevice(const vk::PhysicalDevice &device) {
  vk::PhysicalDeviceProperties2KHR p2;
  vk::PhysicalDevicePCIBusInfoPropertiesEXT pciInfo;
  pciInfo.pNext = p2.pNext;
  p2.pNext = &pciInfo;
  device.getProperties2(&p2);

  for (int cudaDeviceId = 0; cudaDeviceId < 20; cudaDeviceId++)
    if (static_cast<int>(pciInfo.pciBus) ==
        getPCIBusIdFromCudaDeviceId(cudaDeviceId))
      return cudaDeviceId;

  return -1; // should never reach here
}

#endif
