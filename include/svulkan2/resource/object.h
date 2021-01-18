#pragma once
#include "model.h"
#include "svulkan2/common/config.h"
#include "svulkan2/core/allocator.h"
#include <vector>

namespace svulkan2 {
namespace scene {
class Node;
}

namespace resource {

class SVObject {
  std::shared_ptr<StructDataLayout> mBufferLayout;
  std::shared_ptr<SVModel> mModel;
  std::vector<char> mBuffer;
  scene::Node *mParentNode;

  std::unique_ptr<core::Buffer> mDeviceBuffer;

  int mPrevModelMatrixOffset = -1;
  int mModelMatrixOffset = -1;
  int mSegmentationOffset = -1;

  glm::mat4 mPrevModelMatrix{0};
  glm::mat4 mModelMatrix{0};
  glm::uvec4 mSegmentation{0};

  bool mDirty{true};

public:
  SVObject(std::shared_ptr<StructDataLayout> bufferLayout,
           std::shared_ptr<resource::SVModel> model);

  inline void setParentNode(scene::Node *node) { mParentNode = node; };

  void createDeviceResources(core::Context &context);
  void uploadToDevice();

  void setPrevModelMatrix(glm::mat4 const &matrix);
  void setModelMatrix(glm::mat4 const &matrix);
  void setSegmentation(glm::uvec4 const &segmentation);
  template <typename T>
  void setAttribute(std::string const &key, T const &content) {
    auto &elem = mBufferLayout->elements.at(key);
    std::memcpy(mBuffer.data() + elem.offset, &content, elem.size);
  }
};

} // namespace resource
} // namespace svulkan2
