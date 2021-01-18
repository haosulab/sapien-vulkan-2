#ifdef CUDA_INTEROP
#include "svulkan2/core/cuda_buffer.h"
#include "svulkan2/core/context.h"
#include <gtest/gtest.h>

using namespace svulkan2;
using namespace svulkan2::core;

TEST(Buffer, CUDA) {
  Context context(VK_API_VERSION_1_1, false);
  CudaBuffer buffer(context, sizeof(int) * 10,
                    vk::BufferUsageFlagBits::eTransferDst |
                        vk::BufferUsageFlagBits::eTransferSrc);
}
#endif
