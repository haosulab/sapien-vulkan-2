#include "svulkan2/resource/mesh.h"
#include "svulkan2/common/log.h"
#include "svulkan2/core/context.h"
#include <memory>

namespace svulkan2 {
namespace resource {

static void strided_memcpy(void *target, void *source, size_t chunk_size,
                           size_t chunks, size_t stride) {
  char *target_ = reinterpret_cast<char *>(target);
  char *source_ = reinterpret_cast<char *>(source);

  for (size_t i = 0; i < chunks; ++i) {
    std::memcpy(target_, source_, chunk_size);
    target_ += stride;
    source_ += chunk_size;
  }
}

SVMesh::SVMesh() {}

void SVMesh::setIndices(std::vector<uint32_t> const &indices) {
  mDirty = true;
  mIndices = indices;
  mIndexCount = indices.size();
}

std::vector<uint32_t> const &SVMesh::getIndices() const {
  return mIndices;
}

void SVMesh::setVertexAttribute(std::string const &name,
                                std::vector<float> const &attrib) {
  mDirty = true;
  mAttributes[name] = attrib;
}

std::vector<float> const &
SVMesh::getVertexAttribute(std::string const &name) const {
  if (!mAttributes.contains(name)) {
    throw std::runtime_error("attribute " + name + " does not exist on vertex");
  }
  return mAttributes.at(name);
}

bool SVMesh::hasVertexAttribute(std::string const &name) const {
  return mAttributes.contains(name);
}

void SVMesh::uploadToDevice(core::Context &context) {
  if (!mDirty) {
    return;
  }

  auto layout = context.getResourceManager().getVertexLayout();

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
  size_t vertexSize = layout->getSize();
  size_t bufferSize = vertexSize * vertexCount;
  size_t indexBufferSize = sizeof(uint32_t) * mIndices.size();

  std::vector<char> buffer(bufferSize, 0);
  auto elements = layout->getElementsSorted();
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

  if (!mVertexBuffer) {
    mVertexBuffer = std::make_unique<core::Buffer>(
        context, bufferSize,
        vk::BufferUsageFlagBits::eVertexBuffer |
            vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
    mIndexBuffer = std::make_unique<core::Buffer>(
        context, indexBufferSize,
        vk::BufferUsageFlagBits::eIndexBuffer |
            vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  }
  mVertexBuffer->upload(buffer.data(), bufferSize);
  mIndexBuffer->upload<uint32_t>(mIndices);
  mOnDevice = true;
  mDirty = false;
}

void SVMesh::removeFromDevice() {
  mDirty = true;
  mOnDevice = false;
  mVertexBuffer.reset();
  mIndexBuffer.reset();
}

} // namespace resource
} // namespace svulkan2
