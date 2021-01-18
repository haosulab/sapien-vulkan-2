#include "svulkan2/core/buffer.h"
#include "svulkan2/core/context.h"
#include <gtest/gtest.h>

using namespace svulkan2;
using namespace svulkan2::core;

TEST(Buffer, Creation) {
  Context context(VK_API_VERSION_1_1, false);
  Buffer buffer1(context, sizeof(int) * 10,
                 vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eTransferSrc,
                 VMA_MEMORY_USAGE_GPU_ONLY);
  Buffer buffer2(context, sizeof(int) * 10,
                 vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eTransferSrc,
                 VMA_MEMORY_USAGE_CPU_ONLY);
  context.getAllocator().allocateStagingBuffer(sizeof(int) * 100);
}

TEST(Buffer, UploadDownload) {
  Context context(VK_API_VERSION_1_1, false);
  int arr[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  Buffer buffer(context, sizeof(int) * 10,
                vk::BufferUsageFlagBits::eTransferDst |
                    vk::BufferUsageFlagBits::eTransferSrc,
                VMA_MEMORY_USAGE_GPU_ONLY);
  buffer.upload(arr, sizeof(int) * 10, 0);
  int arr2[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  buffer.download(arr2, sizeof(int) * 10, 0);
  for (uint32_t i = 0; i < 10; ++i) {
    ASSERT_EQ(arr2[i], i);
  }

  // copy with offset
  buffer.upload(arr + 1, sizeof(int) * 5,
                sizeof(int) * 5); // 0,1,2,3,4,1,2,3,4,5
  buffer.download(arr2, sizeof(int) * 5, sizeof(int) * 3); // 3,4,1,2,3
  ASSERT_EQ(arr2[0], 3);
  ASSERT_EQ(arr2[1], 4);
  ASSERT_EQ(arr2[2], 1);
  ASSERT_EQ(arr2[3], 2);
  ASSERT_EQ(arr2[4], 3);
}

TEST(Buffer, UploadDownloadVector) {
  Context context(VK_API_VERSION_1_1, false);
  std::vector<int> arr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  Buffer buffer(context, sizeof(int) * 10,
                vk::BufferUsageFlagBits::eTransferDst |
                    vk::BufferUsageFlagBits::eTransferSrc,
                VMA_MEMORY_USAGE_GPU_ONLY);
  buffer.upload<int>(arr);

  auto data2 = buffer.download<int>();
  ASSERT_EQ(data2.size(), 10);
  for (uint32_t i = 0; i < 10; ++i) {
    ASSERT_EQ(data2[i], i);
  }
}
