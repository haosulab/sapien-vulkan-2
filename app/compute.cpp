#include "svulkan2/common/fs.h"
#include "svulkan2/core/context.h"
#include "svulkan2/shader/compute_module.h"
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

  shader::ComputeModule m("../shader/compute/scan.comp");

  auto buffer0 = core::Buffer::Create(sizeof(Buffer0),
                                      vk::BufferUsageFlagBits::eStorageBuffer |
                                          vk::BufferUsageFlagBits::eTransferDst |
                                          vk::BufferUsageFlagBits::eTransferSrc,
                                      VMA_MEMORY_USAGE_GPU_ONLY);
  Buffer0 b0;
  buffer0->upload(b0);

  std::vector<float> data;
  for (uint32_t i = 0; i < 1024; ++i) {
    data.push_back(i);
  }

  auto buffer1 = core::Buffer::Create(sizeof(float) * data.size(),
                                      vk::BufferUsageFlagBits::eStorageBuffer |
                                          vk::BufferUsageFlagBits::eTransferDst |
                                          vk::BufferUsageFlagBits::eTransferSrc,
                                      VMA_MEMORY_USAGE_GPU_ONLY);
  buffer1->upload(data);

  m.bindBuffers({buffer0.get(), buffer1.get()});
  int count = 1024;
  m.bindConstantData(&count, sizeof(int));

  auto pool = context->createCommandPool();
  auto cb = pool->allocateCommandBuffer();

  cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  m.record(cb.get(), (count + 255) / 256);
  cb->end();

  context->getQueue().submitAndWait(cb.get());

  // auto result = buffer1->download<float>();
  // for (auto x : result) {
  //   std::cout << x << " ";
  // }
  // std::cout << std::endl;

  return 0;
}
