#pragma once
#include "svulkan2/common/layout.h"
#include "svulkan2/common/log.h"
#include "svulkan2/common/vk.h"
#include "svulkan2/core/buffer.h"
#include <unordered_map>

namespace svulkan2 {
namespace resource {

class SVMesh {
  std::shared_ptr<InputDataLayout> mVertexLayout;
  std::vector<uint32_t> mIndices;
  std::unordered_map<std::string, std::vector<float>> mAttributes;

  bool mDirty{true};
  bool mOnDevice{false};
  std::unique_ptr<core::Buffer> mVertexBuffer;
  std::unique_ptr<core::Buffer> mIndexBuffer;

public:
  SVMesh(std::shared_ptr<InputDataLayout> layout);

  void setIndices(std::vector<uint32_t> const &indices);
  std::vector<uint32_t> const &getIndices(std::vector<uint32_t>) const;

  void setVertexAttribute(std::string const &name,
                          std::vector<float> const &attrib);
  std::vector<float> const &getVertexAttribute(std::string const &name) const;

  void uploadToDevice(core::Context &context);
  void removeFromDevice();

  inline bool isOnDevice() const { return mOnDevice; }
};
} // namespace resource
} // namespace svulkan2
