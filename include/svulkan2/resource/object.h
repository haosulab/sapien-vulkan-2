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
  std::shared_ptr<SVModel> mModel;
  scene::Node *mParentNode;

  std::unique_ptr<core::Buffer> mDeviceBuffer;

  glm::mat4 mPrevModelMatrix{0};
  glm::mat4 mModelMatrix{0};
  glm::uvec4 mSegmentation{0};

public:
  SVObject( // std::shared_ptr<StructDataLayout> bufferLayout,
      std::shared_ptr<resource::SVModel> model);

  inline void setParentNode(scene::Node *node) { mParentNode = node; };

  // void createDeviceResources(core::Context &context);
  void uploadToDevice(core::Buffer &objectBuffer,
                      StructDataLayout const &objectLayout);

  void setPrevModelMatrix(glm::mat4 const &matrix);
  void setModelMatrix(glm::mat4 const &matrix);

  inline glm::mat4 getPrevModelMatrix() const { return mPrevModelMatrix; };
  inline glm::mat4 getModelMatrix() const { return mModelMatrix; };

  void setSegmentation(glm::uvec4 const &segmentation);
  inline glm::uvec4 getSegmentation() const { return mSegmentation; }
  inline std::shared_ptr<SVModel> getModel() const { return mModel; }
};

} // namespace resource
} // namespace svulkan2
