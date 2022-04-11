#include "svulkan2/core/queue.h"
#include "svulkan2/core/context.h"
#include <easy/profiler.h>

namespace svulkan2 {
namespace core {
Queue::Queue() {
  mContext = Context::Get();
  mQueue = mContext->getDevice().getQueue(
      mContext->getGraphicsQueueFamilyIndex(), 0);
}

void Queue::submit(
    vk::ArrayProxyNoTemporaries<vk::CommandBuffer const> const &commandBuffers,
    vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &waitSemaphores,
    vk::ArrayProxyNoTemporaries<vk::PipelineStageFlags const> const
        &waitStageMasks,
    vk::ArrayProxyNoTemporaries<uint64_t const> const &waitValues,
    vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &signalSemaphores,
    vk::ArrayProxyNoTemporaries<uint64_t const> const &signalValues,
    vk::Fence fence) {
  vk::TimelineSemaphoreSubmitInfo timelineInfo(waitValues, signalValues);
  vk::SubmitInfo info(waitSemaphores, waitStageMasks, commandBuffers,
                      signalSemaphores);
  info.setPNext(&timelineInfo);
  std::lock_guard lock(mMutex);
  mQueue.submit(info, fence);
}

void Queue::submit(
    vk::ArrayProxyNoTemporaries<vk::CommandBuffer const> const &commandBuffers,
    vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &waitSemaphores,
    vk::ArrayProxyNoTemporaries<vk::PipelineStageFlags const> const
        &waitStageMasks,
    vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &signalSemaphores,
    vk::Fence fence) {

  std::vector<vk::CommandBuffer> commandBuffers_(commandBuffers.begin(),
                                                 commandBuffers.end());
  std::vector<vk::Semaphore> waitSemaphores_(waitSemaphores.begin(),
                                             waitSemaphores.end());
  std::vector<vk::Semaphore> signalSemaphores_(signalSemaphores.begin(),
                                               signalSemaphores.end());
  std::vector<vk::PipelineStageFlags> waitStageMasks_(waitStageMasks.begin(),
                                                      waitStageMasks.end());
  std::lock_guard lock(mMutex);
  vk::SubmitInfo info(waitSemaphores_, waitStageMasks_, commandBuffers_,
                      signalSemaphores_);
  mQueue.submit(info, fence);
}

void Queue::submit(
    vk::ArrayProxyNoTemporaries<vk::CommandBuffer const> const &commandBuffers,
    vk::Fence fence) {
  return submit(commandBuffers, {}, {}, {}, fence);
}

vk::Result
Queue::submitAndWait(vk::ArrayProxyNoTemporaries<vk::CommandBuffer const> const
                         &commandBuffers) {
  auto fence = mContext->getDevice().createFenceUnique({});
  submit(commandBuffers, fence.get());
  return mContext->getDevice().waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
}

vk::Result Queue::present(
    vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &waitSemaphores,
    vk::ArrayProxyNoTemporaries<vk::SwapchainKHR const> const &swapchains,
    vk::ArrayProxyNoTemporaries<uint32_t const> const &imageIndices) {
  std::vector<vk::Semaphore> waitSemaphores_(waitSemaphores.begin(),
                                             waitSemaphores.end());
  std::vector<vk::SwapchainKHR> swapchains_(swapchains.begin(),
                                            swapchains.end());
  std::vector<uint32_t> imageIndices_(imageIndices.begin(), imageIndices.end());

  std::lock_guard lock(mMutex);
  vk::PresentInfoKHR info(waitSemaphores_, swapchains_, imageIndices_);
  return mQueue.presentKHR(info);
}

} // namespace core
} // namespace svulkan2
