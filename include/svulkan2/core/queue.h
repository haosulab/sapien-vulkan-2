#pragma once
#include "svulkan2/common/vk.h"
#include <mutex>

namespace svulkan2 {
namespace core {
class Context;

class Queue {
  vk::Queue mQueue;

  std::mutex mMutex;

public:
  Queue();

  void submit(
      vk::ArrayProxyNoTemporaries<vk::CommandBuffer const> const
          &commandBuffers,
      vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &waitSemaphores,
      vk::ArrayProxyNoTemporaries<vk::PipelineStageFlags const> const
          &waitStageMasks,
      vk::ArrayProxyNoTemporaries<uint64_t const> const &waitValues,
      vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &signalSemaphores,
      vk::ArrayProxyNoTemporaries<uint64_t const> const &signalValues,
      vk::Fence fence);

  void submit(
      vk::ArrayProxyNoTemporaries<vk::CommandBuffer const> const
          &commandBuffers,
      vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &waitSemaphores,
      vk::ArrayProxyNoTemporaries<vk::PipelineStageFlags const> const
          &waitStageMasks,
      vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &signalSemaphores,
      vk::Fence fence);

  void submit(vk::ArrayProxyNoTemporaries<vk::CommandBuffer const> const
                  &commandBuffers,
              vk::Fence fence);

  vk::Result
  submitAndWait(vk::ArrayProxyNoTemporaries<vk::CommandBuffer const> const
                    &commandBuffers);

  vk::Result present(
      vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &waitSemaphores,
      vk::ArrayProxyNoTemporaries<vk::SwapchainKHR const> const &swapchains,
      vk::ArrayProxyNoTemporaries<uint32_t const> const &imageIndices);

  inline vk::Queue getVulkanQueue() const { return mQueue; }

  Queue(Queue &) = delete;
  Queue(Queue &&) = delete;
  Queue &operator=(Queue const &) = delete;
  Queue &operator=(Queue &&) = delete;
};

} // namespace core
} // namespace svulkan2
