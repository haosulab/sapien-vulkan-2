#include "svulkan2/ui/window.h"

namespace svulkan2 {
namespace ui {

void Window::build() {
  ImGui::SetNextWindowPos(mPos, ImGuiCond_Once);
  ImGui::SetNextWindowSize(mSize, ImGuiCond_Once);
  ImGui::Begin(mName.c_str());
  for (auto c : mChildren) {
    c->build();
  }
  ImGui::End();
}

} // namespace ui
} // namespace svulkan2
