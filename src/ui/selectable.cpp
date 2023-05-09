#include "svulkan2/ui/selectable.h"
#include <imgui.h>

namespace svulkan2 {
namespace ui {

void Selectable::build() {
  if (ImGui::Selectable(mLabel.c_str(), mSelected) && mCallback) {
    mCallback(std::static_pointer_cast<Selectable>(shared_from_this()));
  }
}
} // namespace ui
} // namespace svulkan2
