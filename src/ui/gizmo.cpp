#include "svulkan2/ui/gizmo.h"

namespace svulkan2 {
namespace ui {

void Gizmo::editTransform() {
  if (ImGui::RadioButton("Translate##gizmoxx",
                         mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
    mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
  ImGui::SameLine();
  if (ImGui::RadioButton("Rotate##gizmoxx",
                         mCurrentGizmoOperation == ImGuizmo::ROTATE))
    mCurrentGizmoOperation = ImGuizmo::ROTATE;
  // ImGui::SameLine();
  // if (ImGui::RadioButton("Scale##gizmoxx",
  //                        mCurrentGizmoOperation == ImGuizmo::SCALE))
  //   mCurrentGizmoOperation = ImGuizmo::SCALE;

  float matrixTranslation[3], matrixRotation[3], matrixScale[3];
  ImGuizmo::DecomposeMatrixToComponents(&mMatrix[0][0], matrixTranslation,
                                        matrixRotation, matrixScale);
  ImGui::InputFloat3("Tr##gizmo", matrixTranslation);
  ImGui::InputFloat3("Rt##gizmo", matrixRotation);
  // ImGui::InputFloat3("Sc##gizmo", matrixScale);
  ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation,
                                          matrixScale, &mMatrix[0][0]);

  if (mCurrentGizmoOperation != ImGuizmo::SCALE) {
    if (ImGui::RadioButton("Local##gizmo",
                           mCurrentGizmoMode == ImGuizmo::LOCAL))
      mCurrentGizmoMode = ImGuizmo::LOCAL;
    ImGui::SameLine();
    if (ImGui::RadioButton("World##gizmo",
                           mCurrentGizmoMode == ImGuizmo::WORLD))
      mCurrentGizmoMode = ImGuizmo::WORLD;
  }

  ImGui::Checkbox("##gizmocheckbox", &mUseSnap);
  ImGui::SameLine();

  glm::vec3 snap;
  switch (mCurrentGizmoOperation) {
  case ImGuizmo::TRANSLATE:
    snap = {0.1, 0.1, 0.1};
    ImGui::InputFloat3("Snap##gizmo", &snap.x);
    break;
  case ImGuizmo::ROTATE:
    snap = {5, 0, 0};
    ImGui::InputFloat("Angle Snap##gizmo", &snap.x);
    break;
  // case ImGuizmo::SCALE:
  //   snap = {0.1, 0, 0};
  //   ImGui::InputFloat("Scale Snap##gizmo", &snap.x);
  //   break;
  }
  ImGuiIO &io = ImGui::GetIO();
  ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
  ImGuizmo::Manipulate(&mView[0][0], &mProjection[0][0], mCurrentGizmoOperation,
                       mCurrentGizmoMode, &mMatrix[0][0], nullptr,
                       mUseSnap ? &snap.x : nullptr);
}

void Gizmo::build() { editTransform(); }

} // namespace ui
} // namespace svulkan2
