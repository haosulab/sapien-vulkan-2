#include "svulkan2/core/mesh.h"
#include <memory>

namespace svulkan2 {
namespace core {

static void strided_memcpy(void *target, void *source, size_t chunk_size,
                           size_t chunks, size_t stride) {
  char *target_ = reinterpret_cast<char *>(target);
  char *source_ = reinterpret_cast<char *>(source);

  for (size_t i = 0; i < chunks; ++i) {
    std::memcpy(target, source, chunk_size);
    target_ += stride;
    source_ += chunk_size;
  }
}

Mesh::Mesh(class Context &context, std::shared_ptr<InputDataLayout> layout)
    : mContext(&context), mVertexLayout(layout) {}

void Mesh::setIndices(std::vector<uint32_t> const &indices) {
  mIndices = indices;
}

void Mesh::setVertexAttribute(std::string const &name,
                              std::vector<float> const &attrib) {
  if (mVertexLayout->elements.find("name") == mVertexLayout->elements.end()) {
    throw std::runtime_error("unknown vertex attribute " + name);
  }
  mAttributes[name] = attrib;
}

void Mesh::upload() {

  if (mAttributes.find("position") == mAttributes.end() ||
      mAttributes["position"].size() == 0) {
    throw std::runtime_error("mesh upload failed: no vertex positions");
  }
  if (!mIndices.size()) {
    throw std::runtime_error("mesh upload failed: empty vertex indices");
  }
  if (mAttributes["position"].size() / 3 * 3 !=
      mAttributes["position"].size()) {
    throw std::runtime_error(
        "mesh upload failed: size of vertex positions is not a multiple of 3");
  }
  if (mIndices.size() / 3 * 3 != mIndices.size()) {
    throw std::runtime_error(
        "mesh upload failed: size of vertex indices is not a multiple of 3");
  }

  size_t vertexCount = mAttributes["position"].size() / 3;
  size_t vertexSize = mVertexLayout->getSize();
  size_t bufferSize = vertexSize * vertexCount;
  size_t indexBufferSize = sizeof(uint32_t) * mIndices.size();

  std::vector<char> buffer(bufferSize, 0);
  auto elements = mVertexLayout->getElementsSorted();
  std::vector<uint32_t> offsets;
  uint32_t offset = 0;
  for (auto &elem : elements) {
    if (mAttributes.find(elem.name) != mAttributes.end()) {
      if (mAttributes[elem.name].size() * sizeof(float) !=
          vertexCount * elem.getSize()) {
        throw std::runtime_error("vertex attribute " + elem.name +
                                 " has incorrect size");
      }
      // TODO: test this
      strided_memcpy(buffer.data() + offset, mAttributes[elem.name].data(),
                     elem.getSize(), vertexCount, vertexSize);
    }
    offset += elem.getSize();
  }
  mVertexBuffer =
      std::make_unique<Buffer>(*mContext, bufferSize,
                               vk::BufferUsageFlagBits::eVertexBuffer |
                                   vk::BufferUsageFlagBits::eTransferDst |
                                   vk::BufferUsageFlagBits::eTransferSrc,
                               VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  mIndexBuffer =
      std::make_unique<Buffer>(*mContext, indexBufferSize,
                               vk::BufferUsageFlagBits::eIndexBuffer |
                                   vk::BufferUsageFlagBits::eTransferDst |
                                   vk::BufferUsageFlagBits::eTransferSrc,
                               VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  mVertexBuffer->upload(buffer.data(), bufferSize);
  mIndexBuffer->upload<uint32_t>(mIndices);
}

} // namespace core
} // namespace svulkan2