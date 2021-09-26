#pragma once
#include "svulkan2/common/glm.h"
#include "widget.h"
#include <ImGuizmo.h>
#include <functional>

namespace svulkan2 {
namespace ui {

UI_CLASS(Gizmo) {
  UI_ATTRIBUTE(Gizmo, std::string, Label);
  UI_ATTRIBUTE(Gizmo, glm::mat4, Matrix);

  ImGuizmo::OPERATION mCurrentGizmoOperation{ImGuizmo::TRANSLATE};
  ImGuizmo::MODE mCurrentGizmoMode{ImGuizmo::WORLD};
  bool mUseSnap{};

  glm::mat4 mView{1};
  glm::mat4 mProjection{1};

  void editTransform();

public:
  void build() override;
  inline void setCameraParameters(glm::mat4 const &view,
                                  glm::mat4 const &projection) {
    mView = view;
    mProjection = projection;
  }
};

} // namespace ui
} // namespace svulkan2
