#include "svulkan2/ui/popup.h"
#include <imgui.h>
#include <string>

namespace svulkan2 {
namespace ui {

void Popup::build() {
  if (!ImGui::IsPopupOpen(getLabelId().c_str())) {
    ImGui::OpenPopup(getLabelId().c_str());
  }

  if (ImGui::BeginPopupModal(getLabelId().c_str(), nullptr)) {
    for (auto c : mChildren) {
      c->build();
    }

    if (ImGui::IsKeyDown(ImGuiKey_Escape)) {
      if (mEscCallback) {
        mEscCallback();
      }
    }

    ImGui::EndPopup();
  }
}

} // namespace ui
} // namespace svulkan2
