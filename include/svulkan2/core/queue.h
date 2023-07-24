#pragma once
#include "svulkan2/common/vk.h"
#include <memory>
#include <mutex>

namespace svulkan2 {
namespace core {
class Device;

class Queue {

public:
  Queue(Device &device, uint32_t familyIndex);

  void submit(vk::ArrayProxyNoTemporaries<vk::CommandBuffer const> const &commandBuffers,
              vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &waitSemaphores,
              vk::ArrayProxyNoTemporaries<vk::PipelineStageFlags const> const &waitStageMasks,
              vk::ArrayProxyNoTemporaries<uint64_t const> const &waitValues,
              vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &signalSemaphores,
              vk::ArrayProxyNoTemporaries<uint64_t const> const &signalValues, vk::Fence fence);

  void submit(vk::ArrayProxyNoTemporaries<vk::CommandBuffer const> const &commandBuffers,
              vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &waitSemaphores,
              vk::ArrayProxyNoTemporaries<vk::PipelineStageFlags const> const &waitStageMasks,
              vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &signalSemaphores,
              vk::Fence fence);

  void submit(vk::ArrayProxyNoTemporaries<vk::CommandBuffer const> const &commandBuffers,
              vk::Fence fence);

  vk::Result
  submitAndWait(vk::ArrayProxyNoTemporaries<vk::CommandBuffer const> const &commandBuffers);

  vk::Result present(vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &waitSemaphores,
                     vk::ArrayProxyNoTemporaries<vk::SwapchainKHR const> const &swapchains,
                     vk::ArrayProxyNoTemporaries<uint32_t const> const &imageIndices);

  inline vk::Queue getVulkanQueue() const { return mQueue; }

  ~Queue();

  Queue(Queue &) = delete;
  Queue(Queue &&) = delete;
  Queue &operator=(Queue const &) = delete;
  Queue &operator=(Queue &&) = delete;

private:
  Device &mDevice;
  vk::Queue mQueue;
  std::mutex mMutex;
};

} // namespace core
} // namespace svulkan2
