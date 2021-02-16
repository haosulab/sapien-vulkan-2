#pragma once
#include "node.h"
#include "svulkan2/resource/model.h"

namespace svulkan2 {
namespace scene {
class Object : public Node {
  std::shared_ptr<resource::SVModel> mModel;
  glm::uvec4 mSegmentation{0};

public:
  inline Type getType() const override { return Type::eObject; }

  Object(std::shared_ptr<resource::SVModel> model,
         std::string const &name = "");

  void uploadToDevice(core::Buffer &objectBuffer,
                      StructDataLayout const &objectLayout);

  void setSegmentation(glm::uvec4 const &segmentation);
  inline glm::uvec4 getSegmentation() const { return mSegmentation; }
  inline std::shared_ptr<resource::SVModel> getModel() const { return mModel; }
};

} // namespace scene
} // namespace svulkan2
