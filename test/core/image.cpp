#include "svulkan2/core/image.h"
#include "svulkan2/common/glm.h"
#include "svulkan2/core/context.h"
#include <gtest/gtest.h>

using namespace svulkan2;
using namespace svulkan2::core;

TEST(Image, Creation) {
  Context context(VK_API_VERSION_1_1, false);
  Image image(context, {4, 2, 1}, vk::Format::eR8G8B8A8Unorm,
              vk::ImageUsageFlagBits::eTransferDst |
                  vk::ImageUsageFlagBits::eTransferSrc,
              VMA_MEMORY_USAGE_GPU_ONLY);
}

TEST(Image, UploadDownload) {
  Context context(VK_API_VERSION_1_1, false);
  Image image(context, {4, 2, 1}, vk::Format::eR8G8B8A8Unorm,
              vk::ImageUsageFlagBits::eTransferDst |
                  vk::ImageUsageFlagBits::eTransferSrc |
                  vk::ImageUsageFlagBits::eColorAttachment |
                  vk::ImageUsageFlagBits::eSampled,
              VMA_MEMORY_USAGE_GPU_ONLY);

  auto cb = context.createCommandBuffer();
  cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  image.transitionLayout(cb.get(), vk::ImageLayout::eUndefined,
                         vk::ImageLayout::eTransferDstOptimal, {},
                         vk::AccessFlagBits::eTransferRead,
                         vk::PipelineStageFlagBits::eTransfer,
                         vk::PipelineStageFlagBits::eTransfer);
  cb->end();
  context.submitCommandBufferAndWait(cb.get());

  std::vector<uint8_t> data = {
      1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1,
      0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
  };
  image.upload<uint8_t>(data);

  cb = context.createCommandBuffer();
  cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  image.transitionLayout(cb.get(), vk::ImageLayout::eShaderReadOnlyOptimal,
                         vk::ImageLayout::eColorAttachmentOptimal,
                         vk::AccessFlagBits::eShaderRead,
                         vk::AccessFlagBits::eColorAttachmentWrite,
                         vk::PipelineStageFlagBits::eFragmentShader,
                         vk::PipelineStageFlagBits::eColorAttachmentOutput);
  cb->end();
  context.submitCommandBufferAndWait(cb.get());

  auto data2 =
      image.download<uint8_t>(vk::Offset3D{0, 0, 0}, vk::Extent3D{4, 2, 1});
  ASSERT_EQ(data2.size(), 32);
  for (uint32_t i = 0; i < 32; ++i) {
    ASSERT_EQ(data[i], data2[i]);
  }

  auto data3 =
      image.download<uint8_t>(vk::Offset3D{1, 1, 0}, vk::Extent3D{2, 1, 1});
  ASSERT_EQ(data3.size(), 8);
  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(data[i + 20], data3[i]);
  }

  auto pixel = image.downloadPixel<uint8_t>(vk::Offset3D{1, 1, 0});
  ASSERT_EQ(pixel.size(), 4);
  ASSERT_EQ(pixel[0], 1);
  ASSERT_EQ(pixel[1], 0);
  ASSERT_EQ(pixel[2], 1);
  ASSERT_EQ(pixel[3], 1);
}
