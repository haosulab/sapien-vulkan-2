#pragma once
#include "svulkan2/common/glm.h"
#include "widget.h"
#include <functional>

namespace svulkan2 {
namespace ui {

UI_CLASS(Gizmo) {
  UI_DECLARE_LABEL(Gizmo);
  UI_ATTRIBUTE(Gizmo, glm::mat4, Matrix);

  UI_BINDING(Gizmo, glm::mat4, Matrix);

  int mCurrentGizmoOperation{7};
  int mCurrentGizmoMode{1};

  bool mUseSnap{};
  glm::vec3 mSnapTranslation{0.1, 0.1, 0.1};
  float mSnapRotation{5};

  glm::mat4 mView{1};
  glm::mat4 mProjection{1};

  void editTransform();

public:
  void build() override;
  inline void setCameraParameters(glm::mat4 const &view, glm::mat4 const &projection) {
    mView = view;
    mProjection = projection;
  }
};

} // namespace ui
} // namespace svulkan2
