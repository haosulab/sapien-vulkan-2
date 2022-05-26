#pragma once
#include "svulkan2/common/layout.h"
#include "svulkan2/common/log.h"
#include "svulkan2/common/vk.h"
#include "svulkan2/core/buffer.h"

namespace svulkan2 {
namespace resource {

class SVPrimitiveSet {
  std::unordered_map<std::string, std::vector<float>> mAttributes;

  bool mOnDevice{false};
  bool mDirty{true};

  uint32_t mVertexCount{};
  std::unique_ptr<core::Buffer> mVertexBuffer;

  std::mutex mUploadingMutex;

public:
  SVPrimitiveSet();
  void setVertexAttribute(std::string const &name,
                          std::vector<float> const &attrib);
  std::vector<float> const &getVertexAttribute(std::string const &name) const;
  core::Buffer &getVertexBuffer();

  size_t getVertexSize();
  inline uint32_t getVertexCount() const { return mVertexCount; }

  void uploadToDevice();
  void removeFromDevice();
};

typedef SVPrimitiveSet SVLineSet;
typedef SVPrimitiveSet SVPointSet;

} // namespace resource
} // namespace svulkan2
