#include "svulkan2/resource/lineset.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace resource {

SVLineSet::SVLineSet() {}

void SVLineSet::setVertexAttribute(std::string const &name,
                                   std::vector<float> const &attrib) {
  mDirty = true;
  mAttributes[name] = attrib;
  if (name == "position") {
    mVertexCount = attrib.size() / 3;
  }
}

std::vector<float> const &
SVLineSet::getVertexAttribute(std::string const &name) const {
  if (mAttributes.find(name) == mAttributes.end()) {
    throw std::runtime_error("attribute " + name + " does not exist on vertex");
  }
  return mAttributes.at(name);
}

void SVLineSet::uploadToDevice(std::shared_ptr<core::Context> context) {
  if (!mDirty) {
    return;
  }
  mContext = context;

  auto layout = context->getResourceManager()->getLineVertexLayout();

  if (mAttributes.find("position") == mAttributes.end() ||
      mAttributes["position"].size() == 0) {
    throw std::runtime_error("lineset upload failed: no vertex positions");
  }
  if (mAttributes["position"].size() / 6 * 6 !=
      mAttributes["position"].size()) {
    throw std::runtime_error(
        "line set upload failed: size of positions is not a multiple of 6");
  }

  size_t vertexCount = mAttributes["position"].size() / 3;
  size_t vertexSize = layout->getSize();
  size_t bufferSize = vertexSize * vertexCount;

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
  }
  mVertexBuffer->upload(buffer.data(), bufferSize);
  mOnDevice = true;
  mDirty = false;
}

void SVLineSet::removeFromDevice() {
  mDirty = true;
  mOnDevice = false;
  mVertexBuffer.reset();
}

} // namespace resource
} // namespace svulkan2
