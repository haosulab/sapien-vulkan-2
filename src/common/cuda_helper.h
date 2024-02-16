#ifdef SVULKAN2_CUDA_INTEROP
#pragma once

#include "svulkan2/common/vk.h"
#include <cuda_runtime.h>
#include <map>
#include <string>

#define checkCudaErrors(call)                                                                     \
  do {                                                                                            \
    cudaError_t err = call;                                                                       \
    if (err != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));  \
      exit(EXIT_FAILURE);                                                                         \
    }                                                                                             \
  } while (0)

inline std::array<uint32_t, 4> parsePCIString(std::string s) {
  if (s.length() == 12) {
    return {static_cast<uint32_t>(std::stoi(s.substr(0, 4), 0, 16)),
            static_cast<uint32_t>(std::stoi(s.substr(5, 2), 0, 16)),
            static_cast<uint32_t>(std::stoi(s.substr(8, 2), 0, 16)),
            static_cast<uint32_t>(std::stoi(s.substr(11, 1), 0, 16))};
  }
  if (s.length() == 7) {
    return {0u, static_cast<uint32_t>(std::stoi(s.substr(0, 2), 0, 16)),
            static_cast<uint32_t>(std::stoi(s.substr(3, 2), 0, 16)),
            static_cast<uint32_t>(std::stoi(s.substr(6, 1), 0, 16))};
  }
  throw std::runtime_error("invalid PCI string");
}

inline std::array<uint32_t, 4> getPCIFromCudaDeviceId(int cudaDeviceId) {
  std::string pciBus(20, '\0');
  if (cudaDeviceGetPCIBusId(pciBus.data(), 20, cudaDeviceId) != cudaSuccess) {
    throw std::runtime_error("invalid cuda device id");
  }
  return parsePCIString(std::string(pciBus.c_str()));
}

inline std::string PCIToString(std::array<uint32_t, 4> pci) {
  std::string pciBus(20, '\0');
  sprintf(pciBus.data(), "%04x:%02x:%02x.%1x", pci[0], pci[1], pci[2], pci[3]);
  return std::string(pciBus.c_str());
}

inline int getCudaDeviceIdFromPhysicalDevice(const vk::PhysicalDevice &device) {

  auto props = device.getProperties2<vk::PhysicalDeviceProperties2,
                                     vk::PhysicalDevicePCIBusInfoPropertiesEXT>();
  std::array<uint32_t, 4> pci{
      props.get<vk::PhysicalDevicePCIBusInfoPropertiesEXT>().pciDomain,
      props.get<vk::PhysicalDevicePCIBusInfoPropertiesEXT>().pciBus,
      props.get<vk::PhysicalDevicePCIBusInfoPropertiesEXT>().pciDevice,
      props.get<vk::PhysicalDevicePCIBusInfoPropertiesEXT>().pciFunction,
  };

  int count;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    return -1;
  }

  for (int cudaDeviceId = 0; cudaDeviceId < count; cudaDeviceId++) {
    if (pci == getPCIFromCudaDeviceId(cudaDeviceId))
      return cudaDeviceId;
  }

  return -1;
}

#endif
