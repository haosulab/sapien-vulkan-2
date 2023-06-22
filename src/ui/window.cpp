#include "svulkan2/ui/window.h"
#include <imgui.h>

namespace svulkan2 {
namespace ui {

void Window::build() {
  ImGui::SetNextWindowPos({mPos.x, mPos.y}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({mSize.x, mSize.y}, ImGuiCond_FirstUseEver);
  ImGui::Begin(getLabelId().c_str());
  for (auto c : mChildren) {
    c->build();
  }
  ImGui::End();
}

} // namespace ui
} // namespace svulkan2
