#include "svulkan2/ui/button.h"
#include <imgui.h>

namespace svulkan2 {
namespace ui {

void Button::build() {
  ImVec2 size{0, 0};
  if (mWidth > 0) {
    size = {mWidth, 0};
  }

  if (ImGui::Button(getLabelId().c_str(), size) && mCallback) {
    mCallback(std::static_pointer_cast<Button>(shared_from_this()));
  }
}
} // namespace ui
} // namespace svulkan2
