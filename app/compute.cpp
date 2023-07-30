#include "svulkan2/common/fs.h"
#include "svulkan2/core/command_buffer.h"
#include "svulkan2/core/context.h"
#include "svulkan2/shader/compute_module.h"
#include <chrono>
#include <iostream>

using namespace svulkan2;

struct Buffer0 {
  int blockCount{0};
  int blockCount2{0};
  float sum{0.f};
};

int main(int argc, char *argv[]) {
  logger::setLogLevel("info");

  auto context = svulkan2::core::Context::Create(true, 5000, 5000, 4);
  auto manager = context->createResourceManager();

  int blockSize = 1024;

  shader::ComputeModule m("../shader/compute/local_scan.comp", blockSize);

  int size = 10000000;

  std::vector<float> data;
  for (uint32_t i = 0; i < size; ++i) {
    data.push_back(i);
  }

  auto data_buffer = core::Buffer::Create(sizeof(float) * data.size(),
                                          vk::BufferUsageFlagBits::eStorageBuffer |
                                              vk::BufferUsageFlagBits::eTransferDst |
                                              vk::BufferUsageFlagBits::eTransferSrc,
                                          VMA_MEMORY_USAGE_GPU_ONLY);
  data_buffer->upload(data);

  auto sum_buffer = core::Buffer::Create(
      sizeof(float) * ((data.size() + blockSize - 1) / blockSize),
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
      VMA_MEMORY_USAGE_GPU_ONLY);

  m.bindBuffers({data_buffer.get(), sum_buffer.get()});
  int count = size;
  m.bindConstantData(&count, sizeof(int));

  auto pool = context->createCommandPool();
  auto cb = pool->allocateCommandBuffer();

  cb->beginOneTime();
  m.record(*cb, (count + blockSize) / blockSize, 1, 1);
  cb->end();

  auto t0 = std::chrono::system_clock().now();
  cb->submitAndWait();
  auto t1 = std::chrono::system_clock().now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
  printf("compute %ld us", us.count());

  auto result = sum_buffer->download<float>();
  for (auto x : result) {
    std::cout << x << " ";
  }
  std::cout << std::endl;

  return 0;
}
