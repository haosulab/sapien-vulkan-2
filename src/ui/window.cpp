#include "svulkan2/ui/window.h"

namespace svulkan2 {
namespace ui {

void Window::build() {
  ImGui::SetNextWindowPos(mPos, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(mSize, ImGuiCond_FirstUseEver);
  ImGui::Begin(mLabel.c_str());
  for (auto c : mChildren) {
    c->build();
  }
  ImGui::End();
}

} // namespace ui
} // namespace svulkan2
