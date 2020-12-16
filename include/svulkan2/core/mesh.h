#pragma once
#include "buffer.h"
#include "svulkan2/common/layout.h"
#include "svulkan2/common/log.h"
#include "svulkan2/common/vk.h"
#include <unordered_map>

namespace svulkan2 {
namespace core {

class Mesh {
  class Context *mContext;

  std::shared_ptr<InputDataLayout> mVertexLayout;
  std::vector<uint32_t> mIndices;
  std::unordered_map<std::string, std::vector<float>> mAttributes;

  std::unique_ptr<Buffer> mVertexBuffer;
  std::unique_ptr<Buffer> mIndexBuffer;

public:
  Mesh(class Context &context, std::shared_ptr<InputDataLayout> layout);

  void setIndices(std::vector<uint32_t> const &indices);
  void setVertexAttribute(std::string const &name,
                          std::vector<float> const &attrib);
  void upload();
};
} // namespace core
} // namespace svulkan2
