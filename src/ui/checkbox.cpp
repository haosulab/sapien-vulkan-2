#include "svulkan2/ui/checkbox.h"
#include <imgui.h>

namespace svulkan2 {
namespace ui {

void Checkbox::build() {
  if (ImGui::Checkbox(mLabel.c_str(), &mChecked) && mCallback) {
    mCallback(std::static_pointer_cast<Checkbox>(shared_from_this()));
  }
}
} // namespace ui
} // namespace svulkan2
