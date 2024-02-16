#include "svulkan2/resource/primitive_set.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace resource {

SVPrimitiveSet::SVPrimitiveSet(uint32_t capacity) : mVertexCapacity(capacity) {}

void SVPrimitiveSet::setVertexAttribute(std::string const &name,
                                        std::vector<float> const &attrib) {
  mDirty = true;
  mAttributes[name] = attrib;
  if (name == "position") {
    mVertexCount = attrib.size() / 3;
  }
  if (this->mOnDevice) {
    this->uploadToDevice();
  }
}

std::vector<float> const &SVPrimitiveSet::getVertexAttribute(std::string const &name) const {
  if (mAttributes.find(name) == mAttributes.end()) {
    throw std::runtime_error("attribute " + name + " does not exist on vertex");
  }
  return mAttributes.at(name);
}

size_t SVPrimitiveSet::getVertexSize() {
  auto layout = core::Context::Get()->getResourceManager()->getLineVertexLayout();
  return layout->getSize();
}

core::Buffer &SVPrimitiveSet::getVertexBuffer() {
  if (!mVertexBuffer) {
    uploadToDevice();
  }
  return *mVertexBuffer;
}

void SVPrimitiveSet::uploadToDevice() {
  std::scoped_lock lock(mUploadingMutex);

  if (!mDirty) {
    return;
  }
  auto context = core::Context::Get();

  auto layout = context->getResourceManager()->getLineVertexLayout();

  if (mAttributes.find("position") == mAttributes.end() || mAttributes["position"].size() == 0) {
    throw std::runtime_error("primitive set upload failed: no vertex positions");
  }

  size_t vertexSize = layout->getSize();

  if (mVertexCapacity == 0) {
    mVertexCapacity = mVertexCount;
  }

  if (mVertexCapacity < mVertexCount) {
    throw std::runtime_error("failed to upload primitive set: vertex count exceeds capacity");
  }

  size_t bufferSize = vertexSize * mVertexCapacity;

  std::vector<char> buffer(bufferSize, 0);
  auto elements = layout->getElementsSorted();
  uint32_t offset = 0;
  for (auto &elem : elements) {
    if (mAttributes.find(elem.name) != mAttributes.end()) {
      if (mAttributes[elem.name].size() * sizeof(float) > mVertexCapacity * elem.getSize()) {
        throw std::runtime_error("vertex attribute " + elem.name + " has incorrect size");
      }
      auto count = mAttributes[elem.name].size() * sizeof(float) / elem.getSize();
      strided_memcpy(buffer.data() + offset, mAttributes[elem.name].data(), elem.getSize(), count,
                     vertexSize);
    }
    offset += elem.getSize();
  }

  // for (auto &elem : elements) {
  //   if (mAttributes.find(elem.name) != mAttributes.end()) {
  //     if (mAttributes[elem.name].size() * sizeof(float) != vertexCount * elem.getSize()) {
  //       throw std::runtime_error("vertex attribute " + elem.name + " has incorrect size");
  //     }
  //     strided_memcpy(buffer.data() + offset, mAttributes[elem.name].data(), elem.getSize(),
  //                    vertexCount, vertexSize);
  //   }
  //   offset += elem.getSize();
  // }

  if (!mVertexBuffer) {
    mVertexBuffer = core::Buffer::Create(
        bufferSize,
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags{}, true);
  }
  mVertexBuffer->upload(buffer.data(), bufferSize);
  mOnDevice = true;
  mDirty = false;
}

void SVPrimitiveSet::removeFromDevice() {
  mDirty = true;
  mOnDevice = false;
  mVertexBuffer.reset();
}

} // namespace resource
} // namespace svulkan2
