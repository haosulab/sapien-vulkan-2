#include "svulkan2/ui/section.h"
#include <imgui.h>

namespace svulkan2 {
namespace ui {

void Section::build() {
  if (ImGui::CollapsingHeader(mLabel.c_str(),
                              mExpanded ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
    for (auto c : mChildren) {
      c->build();
    }
  }
}
} // namespace ui
} // namespace svulkan2
