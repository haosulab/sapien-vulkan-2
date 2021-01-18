#include "svulkan2/core/context.h"
#include <gtest/gtest.h>

using namespace svulkan2;
using namespace svulkan2::core;

TEST(Context, Creation) {
  Context context1(VK_API_VERSION_1_1, false);
  Context context2(VK_API_VERSION_1_1, true);
}

TEST(Context, SubmitCommand) {
  Context context(VK_API_VERSION_1_1, false);
  auto cb = context.createCommandBuffer();
  cb->begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  cb->end();
  context.submitCommandBufferAndWait(cb.get());

  cb = context.createCommandBuffer();
  cb->begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  cb->end();
  context.submitCommandBuffer(cb.get()).wait();

  cb = context.createCommandBuffer();
  cb->begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  cb->end();
  auto fence = context.submitCommandBufferForFence(cb.get());
  context.getDevice().waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
}
